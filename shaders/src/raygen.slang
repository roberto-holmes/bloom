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
RaytracingAccelerationStructure tlas;
[[vk::binding(1)]]
RWTexture2D<float4> read_image;
[[vk::binding(2)]]
RWTexture2D<float4> write_image;

struct CameraUniforms {
	float3 origin;
	float focal_distance;
	float3 u;
	float vfov;
	float3 v;
	float dof_scale;
	float3 w;
}
[[vk::binding(3)]]
cbuffer ubo {
	CameraUniforms camera;
	uint frame_num;
	uint ray_frame_num;
	uint width;
	uint height;
};

struct Payload {
	[[vk::location(0)]]
	float3 radiance;
	float3 attenuation;
	int done;
	float3 ray_origin;
	float3 ray_dir;
};

// if (ubo.ray_frame_num == 10) {
// printf("%f\n", aspect_ratio);
// printf("%v2f : %v2u = %v2f\n", pixelCenter, LaunchSize.xy, uv_in);
// printf("%v3f\n", sample);
// }

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

	Payload payload;
	payload.ray_origin = camera.origin;
	payload.ray_dir = mul(camera_rotation, float3(uv.xy, camera.focal_distance)).xyz;
	payload.attenuation = float3(1.0);
	payload.radiance = float3(0.0);
	payload.done = 0;

	float3 hit_value = 1.0;

	for (int depth = 0; depth < 64; depth++) {
		RayDesc rayDesc;
		rayDesc.Direction = payload.ray_dir;
		rayDesc.Origin = payload.ray_origin;
		rayDesc.TMin = 0.001;
		rayDesc.TMax = 10000.0;

		TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, rayDesc, payload);
		hit_value *= payload.attenuation;

		if (payload.done == 1 || length(payload.attenuation) < 0.1) {
			// if (payload.done == 1) {
			break;
		}
	}
	hit_value *= payload.radiance;

	// float3 sample = (1.0, 1.0, 1.0);
	float3 sample = (0.0, 0.0, 0.0);

	// if (frame_num > 0) {
	// 	sample = read_image[int2(LaunchID.xy)].xyz;
	// }
	sample += hit_value;
	write_image[int2(LaunchID.xy)] = float4(sample, 0.0);
	// write_image[int2(LaunchID.xy)] = float4(float(frame_num) / 100, 0.0, 0.0, 0.0);
}
