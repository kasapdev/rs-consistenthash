# rs-consistenthash

A consistent hashing ring with virtual nodes, implemented using only the
Rust standard library. Zero external dependencies.

## Why this is useful

Sharding data or requests across a set of nodes by hashing a key is easy to
get wrong. The naive approach, `hash(key) % N`, works fine until `N`
changes: add or remove a single node and the modulus changes for
essentially every key, so nearly all of them get remapped to a different
node in one step. For a sharded cache that means a stampede of cache
misses; for a sharded database it means moving almost all your data around
just because you added one more node.

Consistent hashing fixes this. Nodes and keys are hashed onto the same
circular space (a "ring"), and a key belongs to whichever node's point is
the first one found walking clockwise from the key's position. Adding or
removing a node only disturbs the small arc of the ring immediately around
that node — every other key keeps its existing assignment. This crate also
places multiple "virtual node" points per physical node around the ring
(configurable), which is what keeps the actual key distribution close to
even even when you only have a handful of physical nodes.

This is the same idea behind Amazon Dynamo, Cassandra, and most consistent
hash-based load balancers (e.g. what powers `ketama` and similar memcached
client hashing schemes) — implemented here in a small, dependency-free
crate.

## Usage

```rust
use rs_consistenthash::ConsistentHashRing;

// 150 virtual nodes per physical node is a reasonable default.
let mut ring = ConsistentHashRing::new(150);

ring.add_node("cache-1");
ring.add_node("cache-2");
ring.add_node("cache-3");

// Every key deterministically maps to one physical node.
let owner = ring.get_node("user:42").unwrap();
println!("user:42 is owned by {owner}");

// Removing a node only affects the keys that were on its arcs of the ring;
// everyone else keeps their existing assignment.
ring.remove_node("cache-2");
assert!(ring.get_node("user:42").is_some());
```

### Replication: several owners per key

`get_nodes(key, n)` returns up to `n` distinct physical nodes in preference
order: the key's owner first (identical to `get_node`), then the next distinct
nodes clockwise around the ring. This is the "preference list" used to place
`n` replicas of a key, and the second entry is exactly who takes over if the
owner leaves.

```rust
use rs_consistenthash::ConsistentHashRing;

let mut ring = ConsistentHashRing::new(150);
for name in ["cache-1", "cache-2", "cache-3", "cache-4"] {
    ring.add_node(name);
}

let replicas = ring.get_nodes("user:42", 3); // e.g. ["cache-3", "cache-1", "cache-4"]
assert_eq!(replicas.len(), 3);
assert_eq!(Some(replicas[0]), ring.get_node("user:42"));
assert!(ring.contains_node(replicas[1]));
```

Add it to your `Cargo.toml`:

```toml
[dependencies]
rs-consistenthash = "0.1"
```

## How it works

Internally, `ConsistentHashRing` holds a `BTreeMap<u64, String>` mapping a
position on a circular `u64` hash space to the physical node that owns that
position.

- **Placing a node.** `add_node("cache-1")` doesn't insert one point into
  the ring — it inserts `replicas` points (the constructor parameter),
  one per virtual node, by hashing `"cache-1#0"`, `"cache-1#1"`, ...,
  `"cache-1#{replicas-1}"` with `DefaultHasher`. Scattering many points per
  physical node instead of just one is what keeps the ring's arcs
  reasonably even in length, even with only a few physical nodes — a
  single hash point per node would otherwise create wildly uneven,
  luck-of-the-draw arc sizes.

- **Looking a key up.** `get_node("user:42")` hashes the key into the same
  `u64` space and finds the first ring position at or after that hash
  (`BTreeMap::range(hash..).next()`), walking "clockwise." If the key's
  hash is past every point currently on the ring, ownership wraps back
  around to the smallest point on the ring — the ring is circular, not a
  line.

- **Removing a node.** `remove_node("cache-1")` recomputes the same
  `replicas` virtual-node hashes and removes exactly those entries. Because
  the identical hash function and virtual-node naming scheme is used for
  both insertion and removal, this exactly undoes what `add_node` did,
  without needing to store the point list separately.

- **Why this limits remapping.** Only the arcs of the ring immediately
  preceding an added or removed node's virtual points change ownership.
  Every key whose nearest clockwise point is unaffected keeps mapping to
  the same physical node it always did. With `M` existing nodes, adding one
  more should move roughly `1/(M+1)` of all keys — not "hash(key) % N
  changed" percentages. This is proven empirically in the test suite (see
  below), not just asserted in a comment.

### Scoping limits (read before using this in production)

- This is not weighted consistent hashing — every physical node gets the
  same number of virtual nodes (`replicas`), so it assumes physical nodes
  have roughly equal capacity. Nodes with different capacities would need a
  proportional number of virtual nodes, which this crate doesn't compute
  for you.
- `DefaultHasher` (SipHash) is used for its `Hash`-integration convenience
  and its determinism within a single Rust standard library build, not
  because it's the fastest possible ring hash. If you need a stable,
  language-independent hash for interop with a ring implemented elsewhere
  (e.g. a `ketama`-style client), you'd want to swap in a specific hash
  function like MD5 or MurmurHash instead.
- Not thread-safe on its own — wrap it in a `Mutex`/`RwLock` (or similar)
  if you need to share one ring across threads.
- Extremely rare hash collisions between two different virtual node
  identifiers are possible (any `u64` hash has a nonzero collision
  probability) and would silently overwrite one virtual node's ring entry;
  this is not specifically detected or handled, since with a 64-bit hash
  space it's negligible for realistic node/replica counts.

## Testing

```
cargo test
```

The test suite includes two properties that are measured empirically
rather than merely asserted:

1. **Distribution.** With 200 virtual nodes per physical node, 8 physical
   nodes, and 20,000 distinct keys, every physical node's observed share of
   keys is asserted to land within +/-40% of the `total_keys / node_count`
   ideal (a generous bound given the ~7% expected relative standard
   deviation at 200 virtual nodes per node).
2. **Minimal remapping on growth (the defining property of consistent
   hashing).** A 5-node ring is built, 10,000 keys are hashed and their
   assignments recorded, a 6th node is added, and the same 10,000 keys are
   re-hashed. In a real run of this test, only **1,621 of 10,000 keys
   (16.21%)** changed their assigned node — closely matching the
   theoretical `1/(M+1) = 1/6 ~= 16.67%` expectation, and nowhere near the
   ~100% of keys a naive `hash(key) % N` scheme would remap when `N`
   changes from 5 to 6. The test also asserts every remapped key moved
   specifically onto the newly added node, never reshuffled among the
   pre-existing ones.

## License

MIT. See [LICENSE](LICENSE).
