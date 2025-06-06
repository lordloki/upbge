/* SPDX-FileCopyrightText: 2005 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "node_shader_util.hh"

namespace blender::nodes::node_shader_shader_to_rgb_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Shader>("Shader");
  b.add_output<decl::Color>("Color");
  b.add_output<decl::Float>("Alpha");
}

static int node_shader_gpu_shadertorgb(GPUMaterial *mat,
                                       bNode *node,
                                       bNodeExecData * /*execdata*/,
                                       GPUNodeStack *in,
                                       GPUNodeStack *out)
{
  GPU_material_flag_set(mat, GPU_MATFLAG_SHADER_TO_RGBA);

  return GPU_stack_link(mat, node, "node_shader_to_rgba", in, out);
}

}  // namespace blender::nodes::node_shader_shader_to_rgb_cc

/* node type definition */
void register_node_type_sh_shadertorgb()
{
  namespace file_ns = blender::nodes::node_shader_shader_to_rgb_cc;

  static blender::bke::bNodeType ntype;

  sh_node_type_base(&ntype, "ShaderNodeShaderToRGB", SH_NODE_SHADERTORGB);
  ntype.ui_name = "Shader to RGB";
  ntype.ui_description =
      "Convert rendering effect (such as light and shadow) to color. Typically used for "
      "non-photorealistic rendering, to apply additional effects on the output of BSDFs.\nNote: "
      "only supported in EEVEE";
  ntype.enum_name_legacy = "SHADERTORGB";
  ntype.nclass = NODE_CLASS_CONVERTER;
  ntype.declare = file_ns::node_declare;
  ntype.add_ui_poll = object_eevee_shader_nodes_poll;
  ntype.gpu_fn = file_ns::node_shader_gpu_shadertorgb;

  blender::bke::node_register_type(ntype);
}
