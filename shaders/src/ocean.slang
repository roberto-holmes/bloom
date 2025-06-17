__import structs;
__import constants;
__import rng;

static const int IMG_WG_SIZE = 16;

struct PushConstants {
	float wind_speed;
	float wind_angle;
	float leeward_fetch;
	float depth;
	float length_scale;
	uint32_t size;
	uint64_t time_stamp_ns;
}

[[vk::binding(0, 0)]]
RWTexture2D<float4> ocean_map;
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ---------------------------- Base JONSWAP ----------------------------
float jonswap(float dispersion, float dispersion_peak, float speed, float fetch) {
	if (dispersion == 0.0) {
		return 0.0;
	}

	let alpha = intensity(dispersion_peak, speed);

	let beta = 5.0 / 4.0;
	let exponent = -beta * powr(dispersion_peak, 4) / powr(dispersion, 4);

	let gamma = 3.3;

	return ((alpha * g * g) / powr(dispersion, 5)) * exp(exponent) * powr(gamma, jonswap_exponent(dispersion, dispersion_peak));
}

float peak_dispersion(float wind_speed, float fetch) {
	return 22.0 * powr(g * g / (wind_speed * fetch), 0.333333);
}

// A constant that relates to the wind speed and fetch length.
// Typical values in the northern north sea are in the range of 0.0081 to 0.01 (https://www.codecogs.com/library/engineering/fluid_mechanics/waves/spectra/jonswap.php)
float intensity(float wind_speed, float fetch) {
	return 0.076 * powr(wind_speed * wind_speed / (fetch * g), 0.22);
}

float jonswap_exponent(float dispersion, float dispersion_peak) {
	let s = sigma(dispersion, dispersion_peak);

	let num = (dispersion - dispersion_peak) * (dispersion - dispersion_peak);
	let den = 2.0 * dispersion_peak * dispersion_peak * s * s;

	return exp(-(num / den));
}

float sigma(float dispersion, float dispersion_peak) {
	if (dispersion <= dispersion_peak) {
		return 0.07;
	} else {
		return 0.09;
	}
}

// ---------------------------- Kitaigordskii Depth Correction ----------------------------
float depth_correction(float dispersion, float depth) {
	let omega_h = dispersion * sqrt(depth / g);
	if (omega_h <= 1.0) {
		return 0.5 * omega_h * omega_h;
	} else if (omega_h <= 2.0) {
		return 1.0 - 0.5 * (2 - omega_h) * (2 - omega_h);
	} else {
		return 1.0;
	}
}

// ---------------------------- Donelan-Banner Directional Spreading ----------------------------
float directional_spreading(float dispersion, float dispersion_peak, float relative_wind_angle) {
	let ratio = dispersion / dispersion_peak;
	float beta = 0.0;
	if (ratio < 0.56) {
		beta = 2.61 * powr(0.56, 1.3);	// TODO: Verify this is how we want to deal with values outside the equation
	} else if (ratio <= 0.95) {
		beta = 2.61 * powr(ratio, 1.3);
	} else if (ratio <= 1.6) {
		beta = 2.28 * powr(ratio, -1.3);
	} else {
		beta = powr(10.0, -0.4 + 0.8393 * exp(-0.567 * powr(log(ratio), 2)));
	}
	return beta * pow(cosh(beta * relative_wind_angle), -2) / (2.0 * tanh(beta * PI));
}

// ---------------------------- Final Wave Spectrum ----------------------------
float wave_spectrum(float2 wave_vector, float wave_vector_mag, float dispersion, float wind_speed, float fetch, float wind_angle, float depth, float2 delta_k) {
	var relative_wind_angle = (atan2(wave_vector.y, wave_vector.x) - wind_angle);
	if (relative_wind_angle > PI) {
		relative_wind_angle -= 2 * PI;
	} else if (relative_wind_angle < -PI) {
		relative_wind_angle += 2 * PI;
	}
	let dispersion_peak = peak_dispersion(wind_speed, fetch);

	let S_j = jonswap(dispersion, dispersion_peak, wind_speed, fetch);
	let D = directional_spreading(dispersion, dispersion_peak, relative_wind_angle);

	float dw_dk_k = 0.0;

	if (wave_vector_mag != 0.0) {
		dw_dk_k = g / (2 * sqrt(g * wave_vector_mag)) / wave_vector_mag;
	}

	return 2 * S_j * depth_correction(dispersion, depth) * D * dw_dk_k * delta_k.x * delta_k.y;
}

// ---------------------------- Fourier Spectrum ----------------------------
float2 h_0_tilde(float2 wave_vector, float wave_vector_mag, float dispersion, float wind_speed, float fetch, float wind_angle, float depth, float2 delta_k) {
	let common = sqrt(wave_spectrum(wave_vector, wave_vector_mag, dispersion, wind_speed, fetch, wind_angle, depth, delta_k) / 2.0);
	return common * random_unit_vector2();
}

public float2 fourier_spectrum(float2 wave_vector, float wind_speed, float fetch, float wind_angle, float depth, float2 delta_k, float t) {
	let wave_vector_mag = length(wave_vector);
	let dispersion = sqrt(g * wave_vector_mag);

	var base = h_0_tilde(wave_vector, wave_vector_mag, dispersion, wind_speed, fetch, wind_angle, depth, delta_k);
	var conj = h_0_tilde(-wave_vector, wave_vector_mag, dispersion, wind_speed, fetch, wind_angle, depth, delta_k);
	conj.y = -conj.y;

	var x = dispersion * t;
	base *= float2(sin(x), cos(x));
	conj *= float2(sin(-x), cos(-x));

	return base + conj;
}

[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	init_rng(id.xy, pc.size, 0);

	float2 wave_vector = float2(id.xy) / float2(pc.size) * 2 * PI;
	wave_vector /= pc.length_scale;
	float2 delta_k = float2(2 * PI / float(pc.size));

	float time_stamp_s = float(pc.time_stamp_ns) / float(1_000_000_000);
	// if (all(id.xy == uint2(10, 10))) {
	// 	printf("0: %v2f\n", delta_k);
	// 	printf("wind %.6f\n", fourier_spectrum(wave_vector, pc.wind_speed, pc.leeward_fetch, pc.wind_angle, pc.depth, delta_k, time_stamp_s));
	// 	printf("1\n");
	// }
	ocean_map[id.xy] = float4(fourier_spectrum(wave_vector, pc.wind_speed, pc.leeward_fetch, pc.wind_angle, pc.depth, delta_k, time_stamp_s) * 1000000, 0.0, 1.0);
	// ocean_map[id.xy] = float4(1.0, 1.0, 0.0, 1.0);
}
