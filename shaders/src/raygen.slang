/* Copyright (c) 2024, Sascha Willems
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

[shader("raygeneration")]
void main() {
	uint3 LaunchID = DispatchRaysIndex();
	uint3 LaunchSize = DispatchRaysDimensions();

	const float aspect_ratio = LaunchSize.x / LaunchSize.y;

	const float2 pixelCenter = float2(LaunchID.xy) + float2(0.5, 0.5);
	const float2 inUV = pixelCenter / float2(LaunchSize.xy);
	float2 uv = inUV * 2.0 - float2(1.0);
	// float2 uv = mul(2.0 * inUV - float2(1.0), float2(aspect_ratio, -1));
	// float4 target = mul(camera.projInverse, float4(d.x, d.y, 1, 1));

	float viewport_scale_factor = 2. * camera.focal_distance * tan(camera.vfov / 2.);
	// uv *= viewport_scale_factor;

	// float3x3 camera_rotation = float3x3(camera.u, camera.v, camera.w);
	float3x3 camera_rotation = transpose(float3x3(camera.u, camera.v, camera.w));

	RayDesc rayDesc;
	rayDesc.Direction = mul(camera_rotation, float3(uv.xy, camera.focal_distance)).xyz;
	rayDesc.Origin = camera.origin;
	// rayDesc.Origin = mul(camera.viewInverse, float4(0, 0, 0, 1)).xyz;
	// rayDesc.Direction = mul(camera.viewInverse, float4(normalize(target.xyz), 0)).xyz;
	rayDesc.TMin = 0.001;
	rayDesc.TMax = 10000.0;

	Payload payload;
	TraceRay(rs, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, rayDesc, payload);

	// if (LaunchID.y > 100) {
	// 	image[int2(LaunchID.xy)] = float4(0.0, 0.0, 1.0, 0.0);
	// } else {
	image[int2(LaunchID.xy)] = float4(payload.hitValue, 0.0);
	// }
}
