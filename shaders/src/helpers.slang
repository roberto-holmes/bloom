module helpers;

public const static float EPSILON = 1e-2;
public const static float F32_MAX = 3.40282346638528859812e+38;
public const static float PI = 3.14159265359;

public float3 point_on_ray(float t) {
	return WorldRayOrigin() + t * WorldRayDirection();
}
