module helpers;

public float3 point_on_ray(float t) {
	return WorldRayOrigin() + t * WorldRayDirection();
}
