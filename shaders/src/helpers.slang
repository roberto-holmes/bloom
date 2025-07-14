module helpers;

public float3 point_on_ray(float t) {
	return WorldRayOrigin() + t * WorldRayDirection();
}

public float2 expi(float angle) {
	return float2(cos(angle), sin(angle));
}

public float2 cmul(float2 a, float2 b) {
	return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
