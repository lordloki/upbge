/* SPDX-FileCopyrightText: 2021 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edtransform
 */

#include "MEM_guardedalloc.h"

#include "DNA_sequence_types.h"
#include "DNA_space_types.h"

#include "BLI_math_matrix.h"
#include "BLI_math_rotation.h"
#include "BLI_math_vector.h"

#include "SEQ_channels.hh"
#include "SEQ_iterator.hh"
#include "SEQ_relations.hh"
#include "SEQ_sequencer.hh"
#include "SEQ_transform.hh"

#include "ANIM_keyframing.hh"

#include "RNA_access.hh"
#include "RNA_prototypes.hh"

#include "transform.hh"
#include "transform_convert.hh"

namespace blender::ed::transform {

namespace {

/** Used for sequencer transform. */
struct TransDataSeq {
  Strip *strip;
  float orig_origin_position[2];
  float orig_translation[2];
  float orig_scale[2];
  float orig_rotation;
  int orig_flag;
  float active_seq_orig_rotation;
  float2 orig_mirror;
};

}  // namespace

static TransData *SeqToTransData(const Scene *scene,
                                 Strip *strip,
                                 TransData *td,
                                 TransData2D *td2d,
                                 TransDataSeq *tdseq,
                                 int vert_index)
{
  const StripTransform *transform = strip->data->transform;
  const float2 origin = seq::image_transform_origin_offset_pixelspace_get(scene, strip);
  const float2 mirror = seq::image_transform_mirror_factor_get(strip);
  Editing *ed = seq::editing_get(scene);
  float vertex[2] = {origin[0], origin[1]};

  /* Add control vertex, so rotation and scale can be calculated.
   * All three vertices will form a "L" shape that is aligned to the local strip axis.
   */
  if (vert_index == 1) {
    vertex[0] += cosf(transform->rotation);
    vertex[1] += sinf(transform->rotation);
  }
  else if (vert_index == 2) {
    vertex[0] -= sinf(transform->rotation);
    vertex[1] += cosf(transform->rotation);
  }

  td2d->loc[0] = vertex[0];
  td2d->loc[1] = vertex[1];
  td2d->loc2d = nullptr;
  td->loc = td2d->loc;
  copy_v3_v3(td->iloc, td->loc);

  td->center[0] = origin[0];
  td->center[1] = origin[1];

  unit_m3(td->mtx);
  unit_m3(td->smtx);

  axis_angle_to_mat3_single(td->axismtx, 'Z', transform->rotation * mirror[0] * mirror[1]);
  normalize_m3(td->axismtx);

  tdseq->strip = strip;
  copy_v2_v2(tdseq->orig_origin_position, origin);
  tdseq->orig_translation[0] = transform->xofs;
  tdseq->orig_translation[1] = transform->yofs;
  tdseq->orig_scale[0] = transform->scale_x;
  tdseq->orig_scale[1] = transform->scale_y;
  tdseq->orig_rotation = transform->rotation;
  tdseq->orig_flag = strip->flag;
  tdseq->orig_mirror = mirror;
  tdseq->active_seq_orig_rotation = ed->act_strip->data->transform->rotation;

  td->extra = (void *)tdseq;
  td->ext = nullptr;
  td->flag |= TD_SELECTED;
  td->dist = 0.0;

  return td;
}

static void freeSeqData(TransInfo * /*t*/,
                        TransDataContainer *tc,
                        TransCustomData * /*custom_data*/)
{
  TransData *td = tc->data;
  MEM_freeN(td->extra);
}

static void createTransSeqImageData(bContext * /*C*/, TransInfo *t)
{
  Editing *ed = seq::editing_get(t->scene);
  const SpaceSeq *sseq = static_cast<const SpaceSeq *>(t->area->spacedata.first);
  const ARegion *region = t->region;

  if (ed == nullptr) {
    return;
  }
  if (sseq->mainb != SEQ_DRAW_IMG_IMBUF) {
    return;
  }
  if (region->regiontype == RGN_TYPE_PREVIEW && sseq->view == SEQ_VIEW_SEQUENCE_PREVIEW) {
    return;
  }

  ListBase *seqbase = seq::active_seqbase_get(ed);
  ListBase *channels = seq::channels_displayed_get(ed);
  VectorSet strips = seq::query_rendered_strips(t->scene, channels, seqbase, t->scene->r.cfra, 0);
  strips.remove_if([&](Strip *strip) { return (strip->flag & SELECT) == 0; });

  if (strips.is_empty()) {
    return;
  }

  TransDataContainer *tc = TRANS_DATA_CONTAINER_FIRST_SINGLE(t);
  tc->custom.type.free_cb = freeSeqData;

  tc->data_len = strips.size() * 3; /* 3 vertices per sequence are needed. */
  TransData *td = tc->data = MEM_calloc_arrayN<TransData>(tc->data_len, "TransSeq TransData");
  TransData2D *td2d = tc->data_2d = MEM_calloc_arrayN<TransData2D>(tc->data_len,
                                                                   "TransSeq TransData2D");
  TransDataSeq *tdseq = MEM_calloc_arrayN<TransDataSeq>(tc->data_len, "TransSeq TransDataSeq");

  for (Strip *strip : strips) {
    /* One `Sequence` needs 3 `TransData` entries - center point placed in image origin, then 2
     * points offset by 1 in X and Y direction respectively, so rotation and scale can be
     * calculated from these points. */
    SeqToTransData(t->scene, strip, td++, td2d++, tdseq++, 0);
    SeqToTransData(t->scene, strip, td++, td2d++, tdseq++, 1);
    SeqToTransData(t->scene, strip, td++, td2d++, tdseq++, 2);
  }
}

static bool autokeyframe_sequencer_image(bContext *C,
                                         Scene *scene,
                                         StripTransform *transform,
                                         const int tmode)
{
  PropertyRNA *prop;
  PointerRNA ptr = RNA_pointer_create_discrete(&scene->id, &RNA_StripTransform, transform);

  const bool around_cursor = scene->toolsettings->sequencer_tool_settings->pivot_point ==
                             V3D_AROUND_CURSOR;
  const bool do_loc = tmode == TFM_TRANSLATION || around_cursor;
  const bool do_rot = tmode == TFM_ROTATION;
  const bool do_scale = tmode == TFM_RESIZE;
  const bool only_when_keyed = animrig::is_keying_flag(scene, AUTOKEY_FLAG_INSERTAVAILABLE);

  bool changed = false;
  if (do_rot) {
    prop = RNA_struct_find_property(&ptr, "rotation");
    changed |= animrig::autokeyframe_property(
        C, scene, &ptr, prop, -1, scene->r.cfra, only_when_keyed);
  }
  if (do_loc) {
    prop = RNA_struct_find_property(&ptr, "offset_x");
    changed |= animrig::autokeyframe_property(
        C, scene, &ptr, prop, -1, scene->r.cfra, only_when_keyed);
    prop = RNA_struct_find_property(&ptr, "offset_y");
    changed |= animrig::autokeyframe_property(
        C, scene, &ptr, prop, -1, scene->r.cfra, only_when_keyed);
  }
  if (do_scale) {
    prop = RNA_struct_find_property(&ptr, "scale_x");
    changed |= animrig::autokeyframe_property(
        C, scene, &ptr, prop, -1, scene->r.cfra, only_when_keyed);
    prop = RNA_struct_find_property(&ptr, "scale_y");
    changed |= animrig::autokeyframe_property(
        C, scene, &ptr, prop, -1, scene->r.cfra, only_when_keyed);
  }

  return changed;
}

static void recalcData_sequencer_image(TransInfo *t)
{
  TransDataContainer *tc = TRANS_DATA_CONTAINER_FIRST_SINGLE(t);
  TransData *td = nullptr;
  TransData2D *td2d = nullptr;
  Editing *ed = seq::editing_get(t->scene);
  int i;

  for (i = 0, td = tc->data, td2d = tc->data_2d; i < tc->data_len; i++, td++, td2d++) {
    /* Origin. */
    float origin[2];
    copy_v2_v2(origin, td2d->loc);
    i++;
    td++;
    td2d++;

    /* X and Y control points used to read scale and rotation. */
    float handle_x[2];
    copy_v2_v2(handle_x, td2d->loc);
    sub_v2_v2(handle_x, origin);
    i++;
    td++;
    td2d++;

    float handle_y[2];
    copy_v2_v2(handle_y, td2d->loc);
    sub_v2_v2(handle_y, origin);

    TransDataSeq *tdseq = static_cast<TransDataSeq *>(td->extra);
    Strip *strip = tdseq->strip;
    StripTransform *transform = strip->data->transform;

    /* Calculate translation. */
    float translation[2];
    copy_v2_v2(translation, tdseq->orig_origin_position);
    sub_v2_v2(translation, origin);
    mul_v2_v2(translation, tdseq->orig_mirror);
    translation[0] *= t->scene->r.yasp / t->scene->r.xasp;

    /* Round resulting position to integer pixels. Resulting strip
     * will more often end up using faster interpolation (without bilinear),
     * and avoids "text edges are too dark" artifacts with light text strips
     * on light backgrounds. The latter happens because bilinear filtering
     * does not do full alpha pre-multiplication. */
    transform->xofs = roundf(tdseq->orig_translation[0] - translation[0]);
    transform->yofs = roundf(tdseq->orig_translation[1] - translation[1]);

    /* Scale. */
    transform->scale_x = tdseq->orig_scale[0] * fabs(len_v2(handle_x));
    transform->scale_y = tdseq->orig_scale[1] * fabs(len_v2(handle_y));

    /* Rotation. Scaling can cause negative rotation. */
    if (t->mode == TFM_ROTATION) {
      transform->rotation = tdseq->orig_rotation -
                            (t->values_final[0] * tdseq->orig_mirror[0] * tdseq->orig_mirror[1]);
    }

    if (t->mode == TFM_MIRROR) {

      transform->xofs *= t->values_final[0];
      transform->yofs *= t->values_final[1];

      if (t->orient_curr == O_SET) {
        if (strip == ed->act_strip) {
          transform->rotation = -tdseq->orig_rotation;
        }
        else {
          transform->rotation = tdseq->orig_rotation + (2 * -tdseq->active_seq_orig_rotation);
        }
      }
      else {
        strip->flag = tdseq->orig_flag;
        if (t->values_final[0] == -1) {
          strip->flag ^= SEQ_FLIPX;
        }
        if (t->values_final[1] == -1) {
          strip->flag ^= SEQ_FLIPY;
        }
        transform->rotation = tdseq->orig_rotation;
      }
    }

    if ((t->animtimer) && animrig::is_autokey_on(t->scene)) {
      animrecord_check_state(t, &t->scene->id);
      autokeyframe_sequencer_image(t->context, t->scene, transform, t->mode);
    }

    seq::relations_invalidate_cache_preprocessed(t->scene, strip);
  }
}

static void special_aftertrans_update__sequencer_image(bContext * /*C*/, TransInfo *t)
{

  TransDataContainer *tc = TRANS_DATA_CONTAINER_FIRST_SINGLE(t);
  TransData *td = nullptr;
  TransData2D *td2d = nullptr;
  int i;

  for (i = 0, td = tc->data, td2d = tc->data_2d; i < tc->data_len; i++, td++, td2d++) {
    TransDataSeq *tdseq = static_cast<TransDataSeq *>(td->extra);
    Strip *strip = tdseq->strip;
    StripTransform *transform = strip->data->transform;
    if (t->state == TRANS_CANCEL) {
      if (t->mode == TFM_ROTATION) {
        transform->rotation = tdseq->orig_rotation;
      }
      if (t->mode == TFM_MIRROR) {
        transform->xofs = tdseq->orig_translation[0];
        transform->yofs = tdseq->orig_translation[1];
        transform->rotation = tdseq->orig_rotation;
        strip->flag = tdseq->orig_flag;
      }
      continue;
    }

    if (animrig::is_autokey_on(t->scene)) {
      autokeyframe_sequencer_image(t->context, t->scene, transform, t->mode);
    }
  }
}

TransConvertTypeInfo TransConvertType_SequencerImage = {
    /*flags*/ (T_POINTS | T_2D_EDIT),
    /*create_trans_data*/ createTransSeqImageData,
    /*recalc_data*/ recalcData_sequencer_image,
    /*special_aftertrans_update*/ special_aftertrans_update__sequencer_image,
};

}  // namespace blender::ed::transform
