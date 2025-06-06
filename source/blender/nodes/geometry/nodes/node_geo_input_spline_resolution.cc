/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BKE_curves.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_input_spline_resolution_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Int>("Resolution").field_source();
}

class ResolutionFieldInput final : public bke::CurvesFieldInput {
 public:
  ResolutionFieldInput() : bke::CurvesFieldInput(CPPType::get<int>(), "Resolution")
  {
    category_ = Category::NamedAttribute;
  }

  GVArray get_varray_for_context(const bke::CurvesGeometry &curves,
                                 const AttrDomain domain,
                                 const IndexMask & /*mask*/) const final
  {
    return curves.adapt_domain(curves.resolution(), AttrDomain::Curve, domain);
  }

  uint64_t hash() const final
  {
    return 82713465872345682;
  }

  bool is_equal_to(const fn::FieldNode &other) const final
  {
    return dynamic_cast<const ResolutionFieldInput *>(&other) != nullptr;
  }

  std::optional<AttrDomain> preferred_domain(const bke::CurvesGeometry & /*curves*/) const final
  {
    return AttrDomain::Curve;
  }
};

static void node_geo_exec(GeoNodeExecParams params)
{
  params.set_output("Resolution", Field<int>(std::make_shared<ResolutionFieldInput>()));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(
      &ntype, "GeometryNodeInputSplineResolution", GEO_NODE_INPUT_SPLINE_RESOLUTION);
  ntype.ui_name = "Spline Resolution";
  ntype.ui_description =
      "Retrieve the number of evaluated points that will be generated for every control point on "
      "curves";
  ntype.enum_name_legacy = "INPUT_SPLINE_RESOLUTION";
  ntype.nclass = NODE_CLASS_INPUT;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_input_spline_resolution_cc
