module helpers;

__import structs;

public float3 point_on_ray(float t) {
	return WorldRayOrigin() + t * WorldRayDirection();
}

public float2 expi(float angle) {
	return float2(cos(angle), sin(angle));
}

public float2 cmul(float2 a, float2 b) {
	return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

public float get_ocean_point(const Ocean* ocean, float2 coordinates, RWTexture2D<float4> ocean_map) {
	// TODO: Combine with closest_hit_generic::get_ocean_point
	float scaling_factor = ocean.size * ocean.size;

	// Tile coordinates over size
	coordinates %= float2(ocean.size);
	coordinates += float2(ocean.size);
	coordinates %= float2(ocean.size);

	let floored = uint2(floor(coordinates));

	// Bilinear Interpolation
	// TODO: Consider a better algorithm
	// Bottom left
	let h_11 = ocean_map[uint2(floored.x, floored.y + 1) % uint2(ocean.size)] * scaling_factor;
	// Top right
	let h_22 = ocean_map[uint2(floored.x + 1, floored.y) % uint2(ocean.size)] * scaling_factor;
	// Top left
	let h_21 = ocean_map[floored] * scaling_factor;
	// Bottom right
	let h_12 = ocean_map[(floored + uint(1)) % uint2(ocean.size)] * scaling_factor;

	let x_lerp1 = lerp(abs(h_11), abs(h_12), coordinates.x - floored.x);
	let x_lerp2 = lerp(abs(h_21), abs(h_22), coordinates.x - floored.x);
	let y_lerp = lerp(x_lerp2, x_lerp1, coordinates.y - floored.y);

	return y_lerp.x;
}
