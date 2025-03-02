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

struct Payload {
	[[vk::location(0)]]
	float3 radiance;
	float3 attenuation;
	int done;
	float3 ray_origin;
	float3 ray_dir;
};

[shader("miss")]
void main(inout Payload p) {
	// p.radiance = float3(0.4, 0.1, 0.3);
	// p.radiance = float3(0.3, 0.5, 1.0);
	p.radiance = float3(1.0, 1.0, 1.0);
	p.attenuation = float3(1.0, 1.0, 1.0);
	p.done = 1;
}
