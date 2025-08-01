/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edscr
 */

#include <cstdlib>
#include <cstring>

#include "BLI_fileops.h"
#include "BLI_listbase.h"
#include "BLI_path_utils.hh"
#include "BLI_utildefines.h"

#include "BKE_appdir.hh"
#include "BKE_blendfile.hh"
#include "BKE_context.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_screen.hh"
#include "BKE_workspace.hh"

#include "BLO_readfile.hh"

#include "DNA_screen_types.h"
#include "DNA_windowmanager_types.h"
#include "DNA_workspace_types.h"

#include "ED_datafiles.h"
#include "ED_object.hh"
#include "ED_screen.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "UI_interface.hh"
#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "BLT_translation.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "screen_intern.hh"

using blender::Vector;

/* -------------------------------------------------------------------- */
/** \name Workspace API
 *
 * \brief API for managing workspaces and their data.
 * \{ */

WorkSpace *ED_workspace_add(Main *bmain, const char *name)
{
  return BKE_workspace_add(bmain, name);
}

static void workspace_exit(WorkSpace *workspace, wmWindow *win)
{
  /* Scene pinning: Store whatever scene was active when leaving the workspace. It's reactivated
   * when the workspace gets reactivated as well. */
  if (workspace->flags & WORKSPACE_USE_PIN_SCENE) {
    workspace->pin_scene = WM_window_get_active_scene(win);
  }
  else {
    /* The active scene may have been changed. So also always update the unpinned scene to the
     * latest when leaving a workspace that has no scene pinning. */
    win->unpinned_scene = WM_window_get_active_scene(win);
  }
}

/**
 * State changes (old workspace to new workspace):
 * 1) unpinned -> pinned
 *    * Store current scene as the unpinned one (done in #workspace_exit()).
 *    * Change the current scene to the pinned one.
 * 2) pinned   -> pinned
 *    * Change the current scene to the new pinned one.
 * 3) pinned   -> unpinned
 *    * Change current scene back to the unpinned one
 * 4) unpinned -> unpinned
 *    * Make sure the unpinned scene is active.
 *
 * Note that the pin scene must also be updated when leaving a workspace with a pinned scene.
 * That's done separately via workspace_exit() above.
 */
static void workspace_scene_pinning_update(WorkSpace *workspace_new,
                                           const WorkSpace *workspace_old,
                                           bContext *C)
{
  wmWindow *win = CTX_wm_window(C);
  Main *bmain = CTX_data_main(C);
  Scene *active_scene = WM_window_get_active_scene(win);

  const bool is_new_pinned = (workspace_new->flags & WORKSPACE_USE_PIN_SCENE);
  const bool is_old_pinned = (workspace_old->flags & WORKSPACE_USE_PIN_SCENE);

  /* State changes 1 and 2. */
  if (is_new_pinned) {
    if (workspace_new->pin_scene && (workspace_new->pin_scene != active_scene)) {
      WM_window_set_active_scene(bmain, C, win, workspace_new->pin_scene);
      workspace_new->pin_scene = nullptr;
    }
  }
  /* State change 3 - Changing from workspace with pinned scene to unpinned scene. */
  else if (is_old_pinned) {
    if (win->unpinned_scene) {
      WM_window_set_active_scene(bmain, C, win, win->unpinned_scene);
    }
    else {
      /* When leaving a workspace where the pinning was just enabled, the unpinned scene wasn't set
       * yet. */
      win->unpinned_scene = active_scene;
    }
  }
  else {
    /* When leaving a workspace where the pinning was just disabled, we still want to restore the
     * unpinned scene. */
    if (win->unpinned_scene) {
      WM_window_set_active_scene(bmain, C, win, win->unpinned_scene);
    }
  }

  BLI_assert(WM_window_get_active_scene(win));
}

/**
 * Changes the object mode (if needed) to the one set in \a workspace_new.
 * Object mode is still stored on object level. In future it should all be workspace level instead.
 */
static void workspace_change_update(WorkSpace *workspace_new,
                                    WorkSpace *workspace_old,
                                    bContext *C,
                                    wmWindowManager *wm)
{
  workspace_scene_pinning_update(workspace_new, workspace_old, C);
  /* needs to be done before changing mode! (to ensure right context) */
  UNUSED_VARS(wm);
#if 0
  Object *ob_act = CTX_data_active_object(C);
  eObjectMode mode_old = workspace_old->object_mode;
  eObjectMode mode_new = workspace_new->object_mode;

  if (mode_old != mode_new) {
    blender::ed::object::mode_set(C, mode_new);
  }
#endif
}

static WorkSpaceLayout *workspace_change_get_new_layout(Main *bmain,
                                                        WorkSpace *workspace_new,
                                                        wmWindow *win)
{
  WorkSpaceLayout *layout_old = WM_window_get_active_layout(win);
  WorkSpaceLayout *layout_new;

  /* ED_workspace_duplicate may have stored a layout to activate
   * once the workspace gets activated. */
  if (win->workspace_hook->temp_workspace_store) {
    layout_new = win->workspace_hook->temp_layout_store;
  }
  else {
    layout_new = BKE_workspace_active_layout_for_workspace_get(win->workspace_hook, workspace_new);
    if (!layout_new) {
      layout_new = static_cast<WorkSpaceLayout *>(workspace_new->layouts.first);
    }
  }

  return ED_workspace_screen_change_ensure_unused_layout(
      bmain, workspace_new, layout_new, layout_old, win);
}

bool ED_workspace_change(WorkSpace *workspace_new, bContext *C, wmWindowManager *wm, wmWindow *win)
{
  Main *bmain = CTX_data_main(C);
  WorkSpace *workspace_old = WM_window_get_active_workspace(win);
  WorkSpaceLayout *layout_new = workspace_change_get_new_layout(bmain, workspace_new, win);
  bScreen *screen_new = BKE_workspace_layout_screen_get(layout_new);
  bScreen *screen_old = BKE_workspace_active_screen_get(win->workspace_hook);

  win->workspace_hook->temp_layout_store = nullptr;
  if (workspace_old == workspace_new) {
    /* Could also return true, everything that needs to be done was done (nothing :P),
     * but nothing changed */
    return false;
  }

  workspace_exit(workspace_old, win);

  screen_change_prepare(screen_old, screen_new, bmain, C, win);

  if (screen_new == nullptr) {
    return false;
  }

  BKE_workspace_active_layout_set(win->workspace_hook, win->winid, workspace_new, layout_new);
  BKE_workspace_active_set(win->workspace_hook, workspace_new);

  /* update screen *after* changing workspace - which also causes the
   * actual screen change and updates context (including CTX_wm_workspace) */
  screen_change_update(C, win, screen_new);
  workspace_change_update(workspace_new, workspace_old, C, wm);

  BLI_assert(CTX_wm_workspace(C) == workspace_new);

  /* Automatic mode switching. */
  if (workspace_new->object_mode != workspace_old->object_mode) {
    blender::ed::object::mode_set(C, eObjectMode(workspace_new->object_mode));
  }

  return true;
}

WorkSpace *ED_workspace_duplicate(WorkSpace *workspace_old, Main *bmain, wmWindow *win)
{
  WorkSpaceLayout *layout_active_old = BKE_workspace_active_layout_get(win->workspace_hook);
  WorkSpace *workspace_new = ED_workspace_add(bmain, workspace_old->id.name + 2);

  workspace_new->flags = workspace_old->flags;
  workspace_new->pin_scene = workspace_old->pin_scene;
  workspace_new->object_mode = workspace_old->object_mode;
  workspace_new->order = workspace_old->order;
  BLI_duplicatelist(&workspace_new->owner_ids, &workspace_old->owner_ids);

  /* TODO(@ideasman42): tools */

  LISTBASE_FOREACH (WorkSpaceLayout *, layout_old, &workspace_old->layouts) {
    WorkSpaceLayout *layout_new = ED_workspace_layout_duplicate(
        bmain, workspace_new, layout_old, win);

    if (layout_active_old == layout_old) {
      win->workspace_hook->temp_layout_store = layout_new;
    }
  }
  return workspace_new;
}

bool ED_workspace_delete(WorkSpace *workspace, Main *bmain, bContext *C, wmWindowManager *wm)
{
  if (BLI_listbase_is_single(&bmain->workspaces)) {
    return false;
  }

  Vector<ID *> ordered = BKE_id_ordered_list(&bmain->workspaces);
  const int index = ordered.first_index_of(&workspace->id);

  WorkSpace *new_active = reinterpret_cast<WorkSpace *>(index == 0 ? ordered[1] :
                                                                     ordered[index - 1]);

  LISTBASE_FOREACH (wmWindow *, win, &wm->windows) {
    WorkSpace *workspace_active = WM_window_get_active_workspace(win);
    if (workspace_active == workspace) {
      ED_workspace_change(new_active, C, wm, win);
    }
  }

  /* Also delete managed screens if they have no other users. */
  LISTBASE_FOREACH (WorkSpaceLayout *, layout, &workspace->layouts) {
    BKE_id_free_us(bmain, layout->screen);
    layout->screen = nullptr;
  }

  BKE_id_free(bmain, &workspace->id);
  return true;
}

void ED_workspace_scene_data_sync(WorkSpaceInstanceHook *hook, Scene *scene)
{
  bScreen *screen = BKE_workspace_active_screen_get(hook);
  BKE_screen_view3d_scene_sync(screen, scene);
}

/** \} Workspace API */

/* -------------------------------------------------------------------- */
/** \name Workspace Operators
 * \{ */

static WorkSpace *workspace_context_get(bContext *C)
{
  ID *id = UI_context_active_but_get_tab_ID(C);
  if (id && GS(id->name) == ID_WS) {
    return (WorkSpace *)id;
  }

  return CTX_wm_workspace(C);
}

static bool workspace_context_poll(bContext *C)
{
  return workspace_context_get(C) != nullptr;
}

static wmOperatorStatus workspace_new_exec(bContext *C, wmOperator * /*op*/)
{
  Main *bmain = CTX_data_main(C);
  wmWindow *win = CTX_wm_window(C);
  WorkSpace *workspace = workspace_context_get(C);

  workspace = ED_workspace_duplicate(workspace, bmain, win);

  WM_event_add_notifier(C, NC_SCREEN | ND_WORKSPACE_SET, workspace);

  return OPERATOR_FINISHED;
}

static void WORKSPACE_OT_duplicate(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "New Workspace";
  ot->description = "Add a new workspace";
  ot->idname = "WORKSPACE_OT_duplicate";

  /* API callbacks. */
  ot->poll = workspace_context_poll;
  ot->exec = workspace_new_exec;
}

static wmOperatorStatus workspace_delete_exec(bContext *C, wmOperator * /*op*/)
{
  WorkSpace *workspace = workspace_context_get(C);
  WM_event_add_notifier(C, NC_SCREEN | ND_WORKSPACE_DELETE, workspace);
  WM_event_add_notifier(C, NC_WINDOW, nullptr);

  return OPERATOR_FINISHED;
}

static void WORKSPACE_OT_delete(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Delete Workspace";
  ot->description = "Delete the active workspace";
  ot->idname = "WORKSPACE_OT_delete";

  /* API callbacks. */
  ot->poll = workspace_context_poll;
  ot->exec = workspace_delete_exec;
}

static wmOperatorStatus workspace_append_activate_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  char idname[MAX_ID_NAME - 2], filepath[FILE_MAX];

  if (!RNA_struct_property_is_set(op->ptr, "idname") ||
      !RNA_struct_property_is_set(op->ptr, "filepath"))
  {
    return OPERATOR_CANCELLED;
  }
  RNA_string_get(op->ptr, "idname", idname);
  RNA_string_get(op->ptr, "filepath", filepath);

  WorkSpace *appended_workspace = (WorkSpace *)WM_file_append_datablock(
      bmain,
      CTX_data_scene(C),
      CTX_data_view_layer(C),
      CTX_wm_view3d(C),
      filepath,
      ID_WS,
      idname,
      BLO_LIBLINK_APPEND_RECURSIVE);

  if (appended_workspace) {
    if (BLT_translate_new_dataname()) {
      /* Translate workspace name */
      BKE_libblock_rename(
          *bmain, appended_workspace->id, CTX_DATA_(BLT_I18NCONTEXT_ID_WORKSPACE, idname));
    }

    /* Set defaults. */
    BLO_update_defaults_workspace(appended_workspace, nullptr);

    /* Reorder to last position. */
    BKE_id_reorder(&bmain->workspaces, &appended_workspace->id, nullptr, true);

    /* Changing workspace changes context. Do delayed! */
    WM_event_add_notifier(C, NC_SCREEN | ND_WORKSPACE_SET, appended_workspace);

    return OPERATOR_FINISHED;
  }

  return OPERATOR_CANCELLED;
}

static void WORKSPACE_OT_append_activate(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Append and Activate Workspace";
  ot->description = "Append a workspace and make it the active one in the current window";
  ot->idname = "WORKSPACE_OT_append_activate";

  /* API callbacks. */
  ot->exec = workspace_append_activate_exec;

  PropertyRNA *prop;
  RNA_def_string(ot->srna,
                 "idname",
                 nullptr,
                 MAX_ID_NAME - 2,
                 "Identifier",
                 "Name of the workspace to append and activate");
  prop = RNA_def_string(
      ot->srna, "filepath", nullptr, FILE_MAX, "Filepath", "Path to the library");
  RNA_def_property_subtype(prop, PROP_FILEPATH);
  RNA_def_property_flag(prop, PROP_PATH_SUPPORTS_BLEND_RELATIVE);
}

static WorkspaceConfigFileData *workspace_config_file_read(const char *app_template)
{
  const std::optional<std::string> cfgdir = BKE_appdir_folder_id(BLENDER_USER_CONFIG,
                                                                 app_template);
  char startup_file_path[FILE_MAX] = {0};

  if (cfgdir.has_value()) {
    BLI_path_join(
        startup_file_path, sizeof(startup_file_path), cfgdir->c_str(), BLENDER_STARTUP_FILE);
  }

  bool has_path = BLI_exists(startup_file_path);
  return (has_path) ? BKE_blendfile_workspace_config_read(startup_file_path, nullptr, 0, nullptr) :
                      nullptr;
}

static WorkspaceConfigFileData *workspace_system_file_read(const char *app_template)
{
  if (app_template == nullptr) {
    return BKE_blendfile_workspace_config_read(
        nullptr, datatoc_startup_blend, datatoc_startup_blend_size, nullptr);
  }

  char template_dir[FILE_MAX];
  if (!BKE_appdir_app_template_id_search(app_template, template_dir, sizeof(template_dir))) {
    return nullptr;
  }

  char startup_file_path[FILE_MAX];
  BLI_path_join(startup_file_path, sizeof(startup_file_path), template_dir, BLENDER_STARTUP_FILE);

  bool has_path = BLI_exists(startup_file_path);
  return (has_path) ? BKE_blendfile_workspace_config_read(startup_file_path, nullptr, 0, nullptr) :
                      nullptr;
}

static void workspace_append_button(uiLayout *layout,
                                    wmOperatorType *ot_append,
                                    const WorkSpace *workspace,
                                    const Main *from_main)
{
  const ID *id = (ID *)workspace;
  const char *filepath = from_main->filepath;

  if (strlen(filepath) == 0) {
    filepath = BLO_EMBEDDED_STARTUP_BLEND;
  }

  BLI_assert(STREQ(ot_append->idname, "WORKSPACE_OT_append_activate"));

  PointerRNA opptr;
  opptr = layout->op(ot_append,
                     CTX_DATA_(BLT_I18NCONTEXT_ID_WORKSPACE, workspace->id.name + 2),
                     ICON_NONE,
                     blender::wm::OpCallContext::ExecDefault,
                     UI_ITEM_NONE);
  RNA_string_set(&opptr, "idname", id->name + 2);
  RNA_string_set(&opptr, "filepath", filepath);
}

static void workspace_add_menu(bContext * /*C*/, uiLayout *layout, void *template_v)
{
  const char *app_template = static_cast<const char *>(template_v);
  bool has_startup_items = false;

  wmOperatorType *ot_append = WM_operatortype_find("WORKSPACE_OT_append_activate", true);
  WorkspaceConfigFileData *startup_config = workspace_config_file_read(app_template);
  WorkspaceConfigFileData *builtin_config = workspace_system_file_read(app_template);

  if (startup_config) {
    LISTBASE_FOREACH (WorkSpace *, workspace, &startup_config->workspaces) {
      uiLayout *row = &layout->row(false);
      workspace_append_button(row, ot_append, workspace, startup_config->main);
      has_startup_items = true;
    }
  }

  if (builtin_config) {
    bool has_title = false;

    LISTBASE_FOREACH (WorkSpace *, workspace, &builtin_config->workspaces) {
      if (startup_config &&
          BLI_findstring(&startup_config->workspaces, workspace->id.name, offsetof(ID, name)))
      {
        continue;
      }

      if (!has_title) {
        if (has_startup_items) {
          layout->separator();
        }
        has_title = true;
      }

      uiLayout *row = &layout->row(false);
      workspace_append_button(row, ot_append, workspace, builtin_config->main);
    }
  }

  if (startup_config) {
    BKE_blendfile_workspace_config_data_free(startup_config);
  }
  if (builtin_config) {
    BKE_blendfile_workspace_config_data_free(builtin_config);
  }
}

static wmOperatorStatus workspace_add_invoke(bContext *C,
                                             wmOperator *op,
                                             const wmEvent * /*event*/)
{
  uiPopupMenu *pup = UI_popup_menu_begin(
      C, CTX_IFACE_(BLT_I18NCONTEXT_OPERATOR_DEFAULT, op->type->name), ICON_ADD);
  uiLayout *layout = UI_popup_menu_layout(pup);

  layout->menu_fn(IFACE_("General"), ICON_NONE, workspace_add_menu, nullptr);

  ListBase templates;
  BKE_appdir_app_templates(&templates);

  LISTBASE_FOREACH (LinkData *, link, &templates) {
    char *app_template = static_cast<char *>(link->data);
    char display_name[FILE_MAX];

    BLI_path_to_display_name(display_name, sizeof(display_name), IFACE_(app_template));

    /* Steals ownership of link data string. */
    layout->menu_fn_argN_free(display_name, ICON_NONE, workspace_add_menu, app_template);
  }

  BLI_freelistN(&templates);

  layout->separator();
  layout->op("WORKSPACE_OT_duplicate",
             CTX_IFACE_(BLT_I18NCONTEXT_OPERATOR_DEFAULT, "Duplicate Current"),
             ICON_DUPLICATE);

  UI_popup_menu_end(C, pup);

  return OPERATOR_INTERFACE;
}

static void WORKSPACE_OT_add(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Add Workspace";
  ot->description =
      "Add a new workspace by duplicating the current one or appending one "
      "from the user configuration";
  ot->idname = "WORKSPACE_OT_add";

  /* API callbacks. */
  ot->invoke = workspace_add_invoke;
}

static wmOperatorStatus workspace_reorder_to_back_exec(bContext *C, wmOperator * /*op*/)
{
  Main *bmain = CTX_data_main(C);
  WorkSpace *workspace = workspace_context_get(C);

  BKE_id_reorder(&bmain->workspaces, &workspace->id, nullptr, true);
  WM_event_add_notifier(C, NC_WINDOW, nullptr);

  return OPERATOR_INTERFACE;
}

static void WORKSPACE_OT_reorder_to_back(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Workspace Reorder to Back";
  ot->description = "Reorder workspace to be last in the list";
  ot->idname = "WORKSPACE_OT_reorder_to_back";

  /* API callbacks. */
  ot->poll = workspace_context_poll;
  ot->exec = workspace_reorder_to_back_exec;
}

static wmOperatorStatus workspace_reorder_to_front_exec(bContext *C, wmOperator * /*op*/)
{
  Main *bmain = CTX_data_main(C);
  WorkSpace *workspace = workspace_context_get(C);

  BKE_id_reorder(&bmain->workspaces, &workspace->id, nullptr, false);
  WM_event_add_notifier(C, NC_WINDOW, nullptr);

  return OPERATOR_INTERFACE;
}

static void WORKSPACE_OT_reorder_to_front(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Workspace Reorder to Front";
  ot->description = "Reorder workspace to be first in the list";
  ot->idname = "WORKSPACE_OT_reorder_to_front";

  /* API callbacks. */
  ot->poll = workspace_context_poll;
  ot->exec = workspace_reorder_to_front_exec;
}

static wmOperatorStatus workspace_scene_pin_toggle_exec(bContext *C, wmOperator * /*op*/)
{
  WorkSpace *workspace = workspace_context_get(C);

  /* Trivial. The operator is only needed to display a superimposed extra icon, which
   * requires an operator. */
  workspace->flags ^= WORKSPACE_USE_PIN_SCENE;

  WM_event_add_notifier(C, NC_WORKSPACE, nullptr);

  return OPERATOR_FINISHED;
}

static void WORKSPACE_OT_scene_pin_toggle(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Pin Scene to Workspace";
  ot->description =
      "Remember the last used scene for the current workspace and switch to it whenever this "
      "workspace is activated again";
  ot->idname = "WORKSPACE_OT_scene_pin_toggle";

  /* API callbacks. */
  ot->poll = workspace_context_poll;
  ot->exec = workspace_scene_pin_toggle_exec;

  ot->flag = OPTYPE_INTERNAL;
}

void ED_operatortypes_workspace()
{
  WM_operatortype_append(WORKSPACE_OT_duplicate);
  WM_operatortype_append(WORKSPACE_OT_delete);
  WM_operatortype_append(WORKSPACE_OT_add);
  WM_operatortype_append(WORKSPACE_OT_append_activate);
  WM_operatortype_append(WORKSPACE_OT_reorder_to_back);
  WM_operatortype_append(WORKSPACE_OT_reorder_to_front);
  WM_operatortype_append(WORKSPACE_OT_scene_pin_toggle);
}

/** \} Workspace Operators */
