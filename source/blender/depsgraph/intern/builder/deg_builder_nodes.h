/* SPDX-FileCopyrightText: 2013 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup depsgraph
 */

#pragma once

#include "BKE_lib_query.hh" /* For LibraryForeachIDCallbackFlag enum. */

#include "DNA_armature_types.h"
#include "intern/builder/deg_builder.h"
#include "intern/builder/deg_builder_key.h"
#include "intern/builder/deg_builder_map.h"
#include "intern/depsgraph_type.hh"
#include "intern/node/deg_node_id.hh"
#include "intern/node/deg_node_operation.hh"

#include "DEG_depsgraph.hh"

struct BoneCollection;
struct CacheFile;
struct Camera;
struct Collection;
struct FCurve;
struct FreestyleLineSet;
struct FreestyleLineStyle;
struct ID;
struct IDProperty;
struct Image;
struct Key;
struct LayerCollection;
struct Light;
struct LightProbe;
struct ListBase;
struct Main;
struct Mask;
struct Material;
struct MovieClip;
struct Object;
struct ParticleSettings;
struct Scene;
struct Speaker;
struct Tex;
struct VFont;
struct World;
struct bAction;
struct bArmature;
struct bConstraint;
struct bNodeSocket;
struct bNodeTree;
struct bPoseChannel;
struct bSound;
struct PointerRNA;

namespace blender::deg {

struct ComponentNode;
struct Depsgraph;
class DepsgraphBuilderCache;
struct IDNode;
struct OperationKey;
struct OperationNode;
struct TimeSourceNode;

class DepsgraphNodeBuilder : public DepsgraphBuilder {
 public:
  DepsgraphNodeBuilder(Main *bmain, Depsgraph *graph, DepsgraphBuilderCache *cache);
  ~DepsgraphNodeBuilder();

  /* For given original ID get ID which is created by copy-on-evaluation system. */
  ID *get_cow_id(const ID *id_orig) const;
  /* Similar to above, but for the cases when there is no ID node we create
   * one. */
  ID *ensure_cow_id(ID *id_orig);

  /* Helper wrapper function which wraps get_cow_id with a needed type cast. */
  template<typename T> T *get_cow_datablock(const T *orig) const
  {
    return (T *)get_cow_id(&orig->id);
  }

  /* For a given evaluated datablock get corresponding original one. */
  template<typename T> T *get_orig_datablock(const T *cow) const
  {
    return (T *)cow->id.orig_id;
  }

  virtual void begin_build();
  virtual void end_build();

  /**
   * `id_cow_self` is the user of `id_pointer`,
   * see also `LibraryIDLinkCallbackData` struct definition.
   */
  int foreach_id_cow_detect_need_for_update_callback(ID *id_cow_self, ID *id_pointer);

  IDNode *add_id_node(ID *id);
  IDNode *find_id_node(const ID *id);
  TimeSourceNode *add_time_source();

  ComponentNode *add_component_node(ID *id, NodeType comp_type, const char *comp_name = "");
  ComponentNode *find_component_node(const ID *id, NodeType comp_type, const char *comp_name = "");

  OperationNode *add_operation_node(ComponentNode *comp_node,
                                    OperationCode opcode,
                                    const DepsEvalOperationCb &op = nullptr,
                                    const char *name = "",
                                    int name_tag = -1);
  OperationNode *add_operation_node(ID *id,
                                    NodeType comp_type,
                                    const char *comp_name,
                                    OperationCode opcode,
                                    const DepsEvalOperationCb &op = nullptr,
                                    const char *name = "",
                                    int name_tag = -1);
  OperationNode *add_operation_node(ID *id,
                                    NodeType comp_type,
                                    OperationCode opcode,
                                    const DepsEvalOperationCb &op = nullptr,
                                    const char *name = "",
                                    int name_tag = -1);

  OperationNode *ensure_operation_node(ID *id,
                                       NodeType comp_type,
                                       const char *comp_name,
                                       OperationCode opcode,
                                       const DepsEvalOperationCb &op = nullptr,
                                       const char *name = "",
                                       int name_tag = -1);
  OperationNode *ensure_operation_node(ID *id,
                                       NodeType comp_type,
                                       OperationCode opcode,
                                       const DepsEvalOperationCb &op = nullptr,
                                       const char *name = "",
                                       int name_tag = -1);

  bool has_operation_node(ID *id,
                          NodeType comp_type,
                          const char *comp_name,
                          OperationCode opcode,
                          const char *name = "",
                          int name_tag = -1);
  bool has_operation_node(ID *id, NodeType comp_type, OperationCode opcode);

  OperationNode *find_operation_node(const ID *id,
                                     NodeType comp_type,
                                     const char *comp_name,
                                     OperationCode opcode,
                                     const char *name = "",
                                     int name_tag = -1);

  OperationNode *find_operation_node(const ID *id,
                                     NodeType comp_type,
                                     OperationCode opcode,
                                     const char *name = "",
                                     int name_tag = -1);

  OperationNode *find_operation_node(const OperationKey &key);

  virtual void build_id(ID *id, bool force_be_visible = false);

  /* Build function for ID types that do not need their own build_xxx() function. */
  virtual void build_generic_id(ID *id);

  virtual void build_idproperties(IDProperty *id_property);

  virtual void build_scene_render(Scene *scene, ViewLayer *view_layer);
  virtual void build_scene_camera(Scene *scene);
  virtual void build_scene_parameters(Scene *scene);
  virtual void build_scene_compositor(Scene *scene);

  virtual void build_layer_collections(ListBase *lb);
  virtual void build_view_layer(Scene *scene,
                                ViewLayer *view_layer,
                                eDepsNode_LinkedState_Type linked_state);
  virtual void build_collection(LayerCollection *from_layer_collection, Collection *collection);
  virtual void build_object(int base_index,
                            Object *object,
                            eDepsNode_LinkedState_Type linked_state,
                            bool is_visible);
  virtual void build_object_instance_collection(Object *object, bool is_object_visible);
  virtual void build_object_from_layer(int base_index,
                                       Object *object,
                                       eDepsNode_LinkedState_Type linked_state);
  virtual void build_object_flags(int base_index,
                                  Object *object,
                                  eDepsNode_LinkedState_Type linked_state);
  virtual void build_object_modifiers(Object *object);
  virtual void build_object_data(Object *object);
  virtual void build_object_data_camera(Object *object);
  virtual void build_object_data_geometry(Object *object);
  virtual void build_object_data_geometry_datablock(ID *obdata);
  virtual void build_object_data_light(Object *object);
  virtual void build_object_data_lightprobe(Object *object);
  virtual void build_object_data_speaker(Object *object);
  virtual void build_object_data_grease_pencil(Object *object);
  virtual void build_object_transform(Object *object);
  virtual void build_object_constraints(Object *object);
  virtual void build_object_pointcache(Object *object);
  virtual void build_object_shading(Object *object);

  virtual void build_object_light_linking(Object *object);
  virtual void build_light_linking_collection(Collection *collection);

  virtual void build_pose_constraints(Object *object, bPoseChannel *pchan, int pchan_index);
  virtual void build_rigidbody(Scene *scene);
  virtual void build_particle_systems(Object *object, bool is_object_visible);
  virtual void build_particle_settings(ParticleSettings *part);
  /**
   * Build graph nodes for #AnimData block and any animated images used.
   * \param id: ID-Block which hosts the #AnimData
   */
  virtual void build_animdata(ID *id);
  virtual void build_animdata_nlastrip_targets(ListBase *strips);
  /**
   * Build graph nodes to update the current frame in image users.
   */
  virtual void build_animation_images(ID *id);
  virtual void build_action(bAction *action);

  virtual void build_animdata_drivers(ID *id, AnimData *adt);
  /**
   * Build graph node(s) for Driver
   * \param id: ID-Block that driver is attached to
   * \param fcurve: Driver-FCurve
   * \param driver_index: Index in animation data drivers list
   */
  virtual void build_driver(ID *id, FCurve *fcurve, int driver_index);

  virtual void build_driver_variables(ID *id, FCurve *fcurve);
  virtual void build_driver_scene_camera_variable(Scene *scene, const char *camera_path);

  /* Build operations of a property value from which is read by a driver target.
   *
   * The driver target points to a data-block (or a sub-data-block like View Layer).
   * This data-block is presented in the interface as a "Prop" and its resolved RNA pointer is
   * passed here as `target_prop`.
   *
   * The tricky part (and a bit confusing naming) is that the driver target accesses a property of
   * the `target_prop` to get its value. The property which is read to give an actual target value
   * is denoted by its RNA path relative to the `target_prop`. In the interface it is called "Path"
   * and here it is called `rna_path_from_target_prop`. */
  virtual void build_driver_id_property(const PointerRNA &target_prop,
                                        const char *rna_path_from_target_prop);

  virtual void build_parameters(ID *id);
  virtual void build_dimensions(Object *object);
  /** IK Solver Eval Steps. */
  virtual void build_ik_pose(Object *object, bPoseChannel *pchan, bConstraint *con);
  /** Spline IK Eval Steps. */
  virtual void build_splineik_pose(Object *object, bPoseChannel *pchan, bConstraint *con);
  /** Pose/Armature Bones Graph. */
  virtual void build_rig(Object *object);
  virtual void build_armature(bArmature *armature);
  virtual void build_armature_bones(ListBase *bones);
  virtual void build_armature_bone_collections(blender::Span<BoneCollection *> collections);
  /** Shape-keys. */
  virtual void build_shapekeys(Key *key);
  virtual void build_camera(Camera *camera);
  virtual void build_light(Light *lamp);
  virtual void build_nodetree(bNodeTree *ntree);
  virtual void build_nodetree_socket(bNodeSocket *socket);
  /** Recursively build graph for material. */
  virtual void build_material(Material *ma);
  virtual void build_materials(Material **materials, int num_materials);
  virtual void build_freestyle_lineset(FreestyleLineSet *fls);
  virtual void build_freestyle_linestyle(FreestyleLineStyle *linestyle);
  /** Recursively build graph for texture. */
  virtual void build_texture(Tex *tex);
  virtual void build_image(Image *image);
  /** Recursively build graph for world. */
  virtual void build_world(World *world);
  virtual void build_cachefile(CacheFile *cache_file);
  virtual void build_mask(Mask *mask);
  virtual void build_movieclip(MovieClip *clip);
  virtual void build_lightprobe(LightProbe *probe);
  virtual void build_speaker(Speaker *speaker);
  virtual void build_sound(bSound *sound);
  virtual void build_scene_sequencer(Scene *scene);
  virtual void build_scene_audio(Scene *scene);
  virtual void build_scene_speakers(Scene *scene, ViewLayer *view_layer);
  virtual void build_vfont(VFont *vfont);

  /* Per-ID information about what was already in the dependency graph.
   * Allows to re-use certain values, to speed up following evaluation. */
  struct IDInfo {
    /* Copy-on-written pointer of the corresponding ID. */
    ID *id_cow = nullptr;
    /* Mask of visible components from previous state of the
     * dependency graph. */
    IDComponentsMask previously_visible_components_mask = 0;
    /* Special evaluation flag mask from the previous depsgraph. */
    uint32_t previous_eval_flags = 0;
    /* Mesh CustomData mask from the previous depsgraph. */
    DEGCustomDataMeshMasks previous_customdata_masks = {};
  };

 protected:
  /* Entry tags and non-updated operations from the previous state of the dependency graph.
   * The entry tags are operations which were directly tagged, the matching operations from the
   * new dependency graph will be tagged. The needs-update operations are possibly indirectly
   * modified operations, whose complementary part from the new dependency graph will only be
   * marked as needs-update.
   * Stored before the graph is re-created so that they can be transferred over. */
  Vector<PersistentOperationKey> saved_entry_tags_;
  Vector<PersistentOperationKey> needs_update_operations_;

  struct BuilderWalkUserData {
    DepsgraphNodeBuilder *builder;
  };
  static void modifier_walk(void *user_data,
                            struct Object *object,
                            struct ID **idpoin,
                            LibraryForeachIDCallbackFlag cb_flag);
  static void constraint_walk(bConstraint *constraint,
                              ID **idpoin,
                              bool is_reference,
                              void *user_data);

  void tag_previously_tagged_nodes();
  /**
   * Check for IDs that need to be flushed (copy-on-eval-updated)
   * because the depsgraph itself created or removed some of their evaluated dependencies.
   */
  void update_invalid_cow_pointers();

  /* State which demotes currently built entities. */
  Scene *scene_;
  ViewLayer *view_layer_;
  int view_layer_index_;
  /* NOTE: Collection are possibly built recursively, so be careful when
   * setting the current state. */
  /* Accumulated flag over the hierarchy of currently building collections.
   * Denotes whether all the hierarchy from parent of `collection_` to the
   * very root is visible (aka not restricted.). */
  bool is_parent_collection_visible_;

  /* Indexed by original ID.session_uid, values are IDInfo. */
  Map<uint, IDInfo> id_info_hash_;

  /* Set of IDs which were already build. Makes it easier to keep track of
   * what was already built and what was not. */
  BuilderMap built_map_;
};

}  // namespace blender::deg
