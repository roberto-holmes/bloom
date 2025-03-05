
[[vk::binding(0)]]
RaytracingAccelerationStructure rs;
[[vk::binding(1)]]
RWTexture2D<float4> image;

struct CameraUniforms {
	float3 origin;
	float focal_distance;
	float3 u;
	float vfov;
	float3 v;
	float dof_scale;
	float3 w;
}
[[vk::binding(2)]]
cbuffer camera {
	CameraUniforms camera;
};

struct Payload {
	[[vk::location(0)]]
	float3 hitValue;
};

struct Attributes {
	float2 bary;
};

[shader("closesthit")]
void closest_hit(inout Payload p, in Attributes attribs) {
	const float3 barycentricCoords = float3(1.0f - attribs.bary.x - attribs.bary.y, attribs.bary.x, attribs.bary.y);
	p.hitValue = barycentricCoords;
}

[shader("miss")]
void miss(inout Payload p) {
	p.hitValue = float3(0.4, 0.1, 0.3);
}

[shader("raygeneration")]
void ray_gen() {
	uint3 LaunchID = DispatchRaysIndex();
	uint3 LaunchSize = DispatchRaysDimensions();

	const float aspect_ratio = LaunchSize.x / LaunchSize.y;

	const float2 pixelCenter = float2(LaunchID.xy) + float2(0.5, 0.5);
	const float2 inUV = pixelCenter / float2(LaunchSize.xy);
	float2 uv = inUV * 2.0 - float2(1.0);

	float3x3 camera_rotation = transpose(float3x3(camera.u, camera.v, camera.w));

	RayDesc rayDesc;
	rayDesc.Direction = mul(camera_rotation, float3(uv.xy, camera.focal_distance)).xyz;
	rayDesc.Origin = camera.origin;
	rayDesc.TMin = 0.001;
	rayDesc.TMax = 10000.0;

	Payload payload;
	TraceRay(rs, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, rayDesc, payload);

	image[int2(LaunchID.xy)] = float4(payload.hitValue, 0.0);
}
