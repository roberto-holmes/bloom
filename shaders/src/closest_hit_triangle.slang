__import rng;
__import scatter;
__import structs;

[[vk::push_constant]]
ConstantBuffer<RayPushConstants> pc;
// (Binding, Set)
[[vk::binding(0, 2)]]
RWTexture2D<float4> ocean_map[];

[shader("closesthit")]
void main(inout Payload p, in Attributes attribs) {
	init_rng(DispatchRaysIndex().xy, pc.ubo.width, pc.ubo.ray_frame_num + uint(length(WorldRayDirection()) * 100) + pc.ubo.random_num);
	// Get the data corresponding to this object from the Instance Index inserted during TLAS creation
	Mesh* mesh = ((Mesh*)(((Buffers*)pc.scene)[InstanceID()]).primitive);
	let vertices = mesh.vertices;
	let indices = mesh.indices;

	let mat_id = mesh.material_indices[PrimitiveIndex()];
	Material mat = ((Material*)pc.materials)[mat_id];

	// Indices of the triangle
	uint3 ind = uint3(indices[3 * PrimitiveIndex()], indices[3 * PrimitiveIndex() + 1], indices[3 * PrimitiveIndex() + 2]);

	// Vertex of the triangle
	Vertex v0 = vertices[ind.x];
	Vertex v1 = vertices[ind.y];
	Vertex v2 = vertices[ind.z];

	// Barycentric coordinates of the triangle
	const float3 barycentrics = float3(1.0f - attribs.bary.x - attribs.bary.y, attribs.bary.x, attribs.bary.y);

	// Computing the normal at hit position
	float3 normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
	normal = normalize(mul(WorldToObject4x3(), normal).xyz);  // Transforming the normal to world space

	// Computing the coordinates of the hit position
	float3 pos = v0.pos.xyz * barycentrics.x + v1.pos.xyz * barycentrics.y + v2.pos.xyz * barycentrics.z;
	pos = mul(float4(pos, 1.0), ObjectToWorld4x3()).xyz;  // Transforming the position to world space

	scatter(p, pos, normal, mat);
}
