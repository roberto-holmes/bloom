module rng;

struct Rng {
	uint state;
}

static Rng rng_obj;
static const float TAU = 6.283185307;

public void init_rng(uint2 pixel, uint width, uint frame_num) {
	// Seed the PRNG using the scalar index of the pixel and the current frame count.
	uint seed = (pixel.x + pixel.y * width) ^ jenkins_hash(frame_num);
	rng_obj.state = jenkins_hash(seed);
}

// A slightly modified version of the "One-at-a-Time Hash" function by Bob Jenkins.
// See https://www.burtleburtle.net/bob/hash/doobs.html
uint jenkins_hash(uint i) {
	var x = i;
	x += x << 10u;
	x ^= x >> 6u;
	x += x << 3u;
	x ^= x >> 11u;
	x += x << 15u;
	return x;
}

// The 32-bit "xor" function from Marsaglia G., "Xorshift RNGs", Section 3.
uint xorshift32() {
	uint x = rng_obj.state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_obj.state = x;
	return x;
}

// Returns a random float in the range [0...1]. This sets the floating point exponent to zero and
// sets the most significant 23 bits of a random 32-bit unsigned integer as the mantissa. That
// generates a number in the range [1, 1.9999999], which is then mapped to [0, 0.9999999] by
// subtraction. See Ray Tracing Gems II, Section 14.3.4.
public float rand_f32() {
	return asfloat(0x3f800000u | (xorshift32() >> 9u)) - 1.;
}

public float3 generate_random_unit_vector() {
	return normalize(float3(rand_f32() * 2. - 1., rand_f32() * 2. - 1., rand_f32() * 2. - 1.));
	// var v = float3(rand_f32() * 2. - 1., rand_f32() * 2. - 1., rand_f32() * 2. - 1.);
	// while (length(v) > 1.0) {
	// 	v = float3(rand_f32() * 2. - 1., rand_f32() * 2. - 1., rand_f32() * 2. - 1.);
	// };
	// return v;
}

public float3 get_random_in_unit_disk() {
	// var v = float3(rand_f32() * 2. - 1., rand_f32() * 2. - 1., 0.0);
	// while (length(v) > 1.0) {
	// 	v = float3(rand_f32() * 2. - 1., rand_f32() * 2. - 1., 0.0);
	// };
	// return v;
	let r = rand_f32() * 2. - 1.;
	let theta = rand_f32() * TAU;
	return float3(r * cos(theta), r * sin(theta), 0.);
}
