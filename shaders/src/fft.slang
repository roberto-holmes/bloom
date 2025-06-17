// FFT implementation based on https://compute.toys/view/1922 by michael0884
__import constants;

static const int SIZE = 1024;
static const int FFT_WG_SIZE = 128;

static const int RADIX2 = 10;
static const float TWO_PI = 2.0f * PI;

[[vk::binding(0)]]
RWTexture2D<float4> image;

[shader("compute")]
[numthreads(FFT_WG_SIZE, 1, 1)]
void main(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
	// TODO: Figure out how to do a 2D FFT
	// Perform horizontal FFT first
	fft(gtid.x, gid.x, 0, true);
	// GroupMemoryBarrierWithGroupSync();
	// Now do vertical FFT
	// fft(gtid.x, gid.x, 1, true);
	// float4 pixel = float4(1.0, 0.5, 0.5, 1.0);
	// output[id.xy] = pixel;
	// output[id.xy] = float4(id.x / 32, id.y / 32, 0.0, 1.0);
}

uint linearIndex(uint2 id) {
	return id.x + id.y * SIZE;
}

uint2 getAxisIndex(uint id, uint group, uint axis) {
	uint2 idx;
	idx[axis] = id;
	idx[1 - axis] = group;
	return idx;
}

float2 expi(float angle) {
	return float2(cos(angle), sin(angle));
}

float2 cmul(float2 a, float2 b) {
	return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint fftshift(uint index) {
	return (index + SIZE / 2) % SIZE;
}

uint ifftshift(uint index) {
	return (index + (SIZE + 1) / 2) % SIZE;
}

groupshared float2 temp[SIZE];

void radix2(uint span, uint index, bool inverse) {
	// compute pair of indices of elements
	// to perform the radix2 butterfly to
	// every iteration we operate on groups of N * span elements, n our radix
	uint group_size = span << 1;
	uint group_half_mask = span - 1;
	// get the index of this thread relative to group
	uint group_offset = index & group_half_mask;
	// get the index offset of the group this thread is in times two
	uint group_index = (index - group_offset) << 1;
	// first element is group + offset in first group half
	uint k1 = group_index + group_offset;
	// second element is group + offset in second group half
	uint k2 = k1 + span;

	float d = inverse ? 1.0 : -1.0;
	float angle = TWO_PI * d * float(group_offset) / float(group_size);

	// radix2 butterfly
	float2 v1 = temp[k1];
	float2 v2 = cmul(expi(angle), temp[k2]);
	temp[k1] = v1 + v2;
	temp[k2] = v1 - v2;
}

void fft(uint index, uint group, uint axis, bool inverse) {
	// number of elements to load per workgroup thread
	uint M = SIZE / FFT_WG_SIZE;

	// load elements from input buffer and store them at bit reversed indices
	for (uint i = 0u; i < M; i++) {
		uint rowIndex = index + i * FFT_WG_SIZE;
		uint idx = reversebits(rowIndex) >> (32u - RADIX2);
		temp[idx] = image[getAxisIndex(rowIndex, group, axis)].rg;
	}

	// wait for data be loaded
	GroupMemoryBarrierWithGroupSync();

	// in-place FFT loop
	for (uint span = 1u; span < SIZE; span *= 2u) {
		for (uint j = 0u; j < (M >> 1); j++) {
			let rowIndex = index + j * FFT_WG_SIZE;
			radix2(span, rowIndex, inverse);
		}
		// wait for all warps to complete work
		GroupMemoryBarrierWithGroupSync();
	}

	// store the result back into input buffer
	for (uint i = 0u; i < M; i++) {
		uint rowIndex = index + i * FFT_WG_SIZE;
		uint2 idx = getAxisIndex(rowIndex, group, axis);
		image[idx].rg = temp[rowIndex] / (inverse ? SIZE : 1.0);
	}
}
