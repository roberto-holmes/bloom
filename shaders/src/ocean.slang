__import structs;
__import constants;
__import rng;

[[vk::binding(6)]]
RWTexture2D<float4> ocean_map;

public float4 hit_ocean(const Ocean* ocean) {
	float t_max = 10.0;
	float dt = 0.01;
	// Ray march to see if we hit the ocean
	// TODO: Optimise
	for (float t = 0.0; t < t_max; t += dt) {
		let p = WorldRayOrigin() + WorldRayDirection() * t;
		let point = get_ocean_point(ocean, p.xz);
		if (p.y < point.r) {
			return point;
		}
	}
	return float4(F32_MAX);
}

float4 get_ocean_point(const Ocean* ocean, float2 coordinates) {
	// Tile coordinates over size
	coordinates -= ocean.size * floor(coordinates / ocean.size);
	// Bilinear Interpolation
	// TODO: Verify interpolation implementation
	// TODO: Consider a better algorithm
	let h_11 = ocean_map[uint2(floor(coordinates))];
	let h_22 = ocean_map[uint2(ceil(coordinates))];
	let h_21 = ocean_map[uint2((uint)floor(coordinates.x), (uint)ceil(coordinates.y))];
	let h_12 = ocean_map[uint2((uint)ceil(coordinates.x), (uint)floor(coordinates.y))];

	let x_lerp1 = lerp(h_11, h_12, ceil(coordinates.x) - floor(coordinates.x));
	let x_lerp2 = lerp(h_21, h_22, ceil(coordinates.x) - floor(coordinates.x));
	let y_lerp1 = lerp(h_11, h_21, ceil(coordinates.y) - floor(coordinates.y));
	let y_lerp2 = lerp(h_12, h_22, ceil(coordinates.y) - floor(coordinates.y));

	let x_lerp = lerp(x_lerp1, x_lerp2, ceil(coordinates.x) - floor(coordinates.x));
	let y_lerp = lerp(y_lerp1, y_lerp2, ceil(coordinates.y) - floor(coordinates.y));

	return (x_lerp + y_lerp) / 2.0;
}

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
