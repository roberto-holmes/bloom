use crate::api::{Bloomable, Child, Collider, Instance, Orientation};
use hecs::World;
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

pub fn system<T: Bloomable>(user: &mut T, _delta: Duration, world: &Arc<RwLock<World>>) {
    // let mut w = world.write().unwrap();

    // // Make sure that all transformations matrices are calculated and up to date
    // for o in w.query_mut::<&mut Orientation>().view().iter_mut() {
    //     o.1.update();
    // }

    // evaluate_collisions(user, &mut w);
    // // Update all parents
    // evaluate_relative_transforms(&mut w);
}

fn evaluate_collisions<T: Bloomable>(user: &mut T, w: &mut World) {
    let cam = match user.get_active_camera() {
        Some(c) => c,
        None => return,
    };
    let player_entity = match w.query_one::<&Child>(cam).unwrap().get() {
        Some(v) => v.parent,
        None => return, // We don't need to do physics if the camera is not attached to anything
    };
    let mut player_query = w
        .query_one::<(&mut Collider, &mut Orientation, &Instance)>(player_entity)
        .unwrap();
    let player = player_query.get().unwrap();

    for (c, (c_coll, c_ori, c_inst)) in w
        .query::<(&Collider, &Orientation, &Instance)>()
        .without::<&bool>() // We need this bodge so we can avoid the player and not create two references to it
        .view()
        .iter_mut()
    {
        if c == player_entity {
            // Don't check collisions with itself
            continue;
        }
        // Check for collision
        if player
            .0
            .aabb
            .apply(player.1.transformation * player.2.base_transform)
            .collides(
                &c_coll
                    .aabb
                    .apply(c_ori.transformation * c_inst.base_transform),
            )
        {
            // Move back to last good position
            player.1.pos = player.0.last_pos;
            player.1.update();
        }
    }
    player.0.last_pos = player.1.pos;
}

/// Update absolute transforms based on relative transforms (from hecs examples)
fn evaluate_relative_transforms(world: &mut World) {
    // Construct a view for efficient random access into the set of all entities that have
    // parents. Views allow work like dynamic borrow checking or component storage look-up to be
    // done once rather than per-entity as in `World::get`.
    let mut children = world.query::<&Child>();
    let children = children.view();

    // View of entities that aren't children, i.e. roots of the transform hierarchy
    let mut roots = world.query::<&Orientation>().without::<&Child>();
    let roots = roots.view();

    // This query can coexist with the `roots` view without illegal aliasing of `Orientation`
    // references because the inclusion of `&Child` in the query, and its exclusion from the view,
    // guarantees that they will never overlap. Similarly, it can coexist with `parents` because
    // that view does not reference `Orientation`s at all.
    for (_entity, (child, absolute)) in world.query::<(&Child, &mut Orientation)>().iter() {
        // Walk the hierarchy from this entity to the root, accumulating the entity's absolute
        // transform. This does a small amount of redundant work for intermediate levels of deeper
        // hierarchies, but unlike a top-down traversal, avoids tracking entity child lists and is
        // cache-friendly.
        let mut relative_quat = child.offset_quat;
        let mut relative_pos = child.offset_pos;
        let mut ancestor = child.parent;
        while let Some(next) = children.get(ancestor) {
            relative_pos = next.offset_pos + relative_pos;
            relative_quat = next.offset_quat * relative_quat;
            ancestor = next.parent;
        }
        // The `while` loop terminates when `ancestor` cannot be found in `children`, i.e. when it
        // does not have a `Child` component, and is therefore necessarily a root.
        absolute.pos = roots.get(ancestor).unwrap().pos
            + roots.get(ancestor).unwrap().quat.apply(relative_pos);
        // absolute.pos = roots.get(ancestor).unwrap().pos + relative_quat.apply(relative_pos);
        absolute.quat = roots.get(ancestor).unwrap().quat * relative_quat;
        absolute.update();
    }
}
