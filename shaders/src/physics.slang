__import structs;
__import constants;
__import rng;
__import helpers;

struct Instance {
	AccelerationStructureInstanceKHR asi;
	uint32_t id;
}

struct AccelerationStructureInstanceKHR {
	float4 row1;
	float4 row2;
	float4 row3;
	uint32_t instanceMask;
	uint32_t res2;
	uint32_t res3;
	uint32_t res4;
}

struct PushConstants {
	uint64_t time_stamp_ns;
	uint64_t instances;
	// StructuredBuffer<AccelerationStructureInstanceKHR>* instances;
	uint32_t instance_count;
}

// [[vk::binding(0, 0)]]
// RWTexture2D<float4> ocean_map;	// TODO: Consider replacing with a float2 texture
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
// [numthreads(8, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	if (all(id.xy == uint2(10, 10))) {
		// if (id.x == 10) {
		// StructuredBuffer<AccelerationStructureInstanceKHR> instance_buffer = *pc.instances;
		// printf("%u instances, addr: %u, 1st: %u\n", pc.instance_count, pc.instances, *((uint32_t*)pc.instances));

		// uint32_t4* p32 = (uint32_t4*)pc.instances;
		// uint64_t2* p64 = (uint64_t2*)pc.instances;

		// // for (int i = 0; i < pc.instance_count; i++) {
		// printf("32 - 1: %u %u %u %u\n", *((uint32_t*)pc.instances), *((uint32_t*)pc.instances + 1), *((uint32_t*)pc.instances + 2), *((uint32_t*)pc.instances + 3));
		// printf("64 - 1: %u %u\n", *((uint64_t*)pc.instances), *((uint64_t*)pc.instances + 1));

		// printf("32v - 1: %v4u\n", *p32);
		// printf("32v - 2: %v4u\n", *(p32 + 1));

		// printf("64v - 1: %v2u\n", *p64);
		// printf("64v - 2: %v2u\n", *(p64 + 1));

		// printf("32:%u - 1: %v4u\n", i, *(uint32_t4*)pc.instances);
		// printf("32:%u - 2: %v4u\n", i, *((uint32_t4*)pc.instances + 1));
		// printf("32:%u - 3: %v4u\n", i, *((uint32_t4*)pc.instances + 2));
		// printf("32:%u - 4: %v4u\n", i, *((uint32_t4*)pc.instances + 3));

		// printf("64:%u - 1: %v2u\n", i, *(uint64_t2*)pc.instances);
		// printf("64:%u - 2: %v2u\n", i, *((uint64_t2*)pc.instances + 1));
		// printf("64:%u - 3: %v2u\n", i, *((uint64_t2*)pc.instances + 2));
		// printf("64:%u - 4: %v2u\n", i, *((uint64_t2*)pc.instances + 3));
		// }

		// for (int i = 0; i < pc.instance_count; i++) {
		// 	printf("64:%u - 1: %v2u\n", i, *p64);
		// 	printf("64:%u - 2: %v2u\n", i, *(p64 + 1));
		// 	printf("64:%u - 3: %v2u\n", i, *(p64 + 2));
		// 	printf("64:%u - 4: %v2u\n", i, *(p64 + 3));
		// 	p64 = p64 + 4;

		// printf("64:%u - 1: %v2u\n", i, *(uint64_t2*)pc.instances);
		// printf("64:%u - 2: %v2u\n", i, *((uint64_t2*)pc.instances + 1));
		// printf("64:%u - 3: %v2u\n", i, *((uint64_t2*)pc.instances + 2));
		// printf("64:%u - 4: %v2u\n", i, *((uint64_t2*)pc.instances + 3));
		// }
		// printf("%u instances, top: %v4f\n", pc.instance_count, *(float4*)pc.instances);
		// printf("1: %v4f\n", (*(Instance*)*(uint64_t*)pc.instances).asi.row1);
		// printf("2: %v4f\n", (*(Instance*)*(uint64_t*)pc.instances).asi.row2);
		// printf("3: %v4f\n", (*(Instance*)*(uint64_t*)pc.instances).asi.row3);
		// printf("1: %v4f\n", *(float4*)*(uint64_t*)pc.instances);
		// printf("2: %v4f\n", *(float4*)((*(uint64_t*)pc.instances) + 1));
		// printf("3: %v4f\n", *(float4*)((*(uint64_t*)pc.instances) + 2));
		// printf("4: %v4f\n", *(float4*)((*(uint64_t*)pc.instances) + 3));
		// printf("1: %v4f\n", instance_buffer[0].row1);
		// printf("2: %v4f\n", instance_buffer[0].row2);
		// printf("3: %v4f\n", instance_buffer[0].row3);

		// TODO: Figure out how to identify entity
		(*(Instance*)*(uint64_t*)pc.instances).asi.row1 = float4(1.0, 0.0, 0.0, pc.time_stamp_ns / float(100_000_000));

		if (pc.time_stamp_ns < 50_000_000) {
			printf("Moving entity %u\n", (*(Instance*)*(uint64_t*)pc.instances).id);
		}
	}
}
