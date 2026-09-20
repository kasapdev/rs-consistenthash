//! # rs-consistenthash
//!
//! A consistent hashing ring with virtual nodes, built entirely on the Rust
//! standard library (no external dependencies).
//!
//! Consistent hashing solves a very specific, very common problem: when you
//! shard data (or requests) across a set of nodes by hashing a key, adding
//! or removing a node with plain `hash(key) % N` remaps *almost every* key,
//! because `N` changed. That's catastrophic for caches, sharded databases,
//! or load balancers, where remapping a key usually means a cache miss or a
//! costly data migration.
//!
//! Consistent hashing fixes this by hashing both nodes and keys onto the
//! same circular space (a "ring"). A key belongs to the first node found by
//! walking clockwise from the key's position. When a node is added or
//! removed, only the keys that fall between that node and its predecessor
//! on the ring move — everyone else's assignment is untouched. See
//! [`ConsistentHashRing`] for the implementation, and the crate tests for an
//! empirical proof of this property.
//!
//! ## Example
//!
//! ```
//! use rs_consistenthash::ConsistentHashRing;
//!
//! let mut ring = ConsistentHashRing::new(100);
//! ring.add_node("server-a");
//! ring.add_node("server-b");
//! ring.add_node("server-c");
//!
//! // Every key deterministically maps to one of the physical nodes.
//! let node = ring.get_node("user:12345").unwrap();
//! assert!(["server-a", "server-b", "server-c"].contains(&node));
//!
//! // The very same key always resolves to the very same node, as long as
//! // the ring's membership hasn't changed.
//! assert_eq!(ring.get_node("user:12345"), ring.get_node("user:12345"));
//! ```

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};

/// A consistent hashing ring that distributes keys across a set of named
/// physical nodes using virtual nodes for smoother distribution.
///
/// # How it works
///
/// Internally the ring is a [`BTreeMap<u64, String>`] mapping a position on
/// a circular `u64` hash space to the physical node that "owns" that
/// position. Each physical node is not placed once but `replicas` times —
/// once per **virtual node** — by hashing `"{name}#{i}"` for
/// `i in 0..replicas`. This spreads each physical node's ownership across
/// many small, scattered arcs of the ring instead of one single arc, which
/// is what makes the distribution of keys across nodes reasonably even even
/// when there are only a handful of physical nodes (a handful of raw hash
/// points would otherwise create wildly uneven arc lengths).
///
/// To look a key up, the key is hashed to the same `u64` space, and the
/// ring is walked clockwise (i.e. towards larger `u64` values) to find the
/// first virtual node point at or after the key's position, wrapping back
/// around to the smallest point on the ring if the key hashes past the
/// largest point. That owning virtual node's physical node is the answer.
///
/// Because both nodes and keys share the same hash space, adding or
/// removing a physical node only perturbs the small arcs immediately
/// surrounding that node's virtual points — every other key keeps mapping
/// to the same physical node it always did. This is the defining property
/// of consistent hashing, proven empirically in this crate's test suite.
pub struct ConsistentHashRing {
    /// Number of virtual nodes placed on the ring per physical node.
    replicas: usize,
    /// Ring position (hash) -> physical node name.
    ring: BTreeMap<u64, String>,
    /// The set of physical node names currently present in the ring, kept
    /// so `add_node`/`remove_node` can be idempotent and so we know exactly
    /// which virtual-node hashes to remove without recomputing membership.
    nodes: HashSet<String>,
}

impl ConsistentHashRing {
    /// Creates a new, empty consistent hashing ring.
    ///
    /// `replicas` is the number of virtual nodes placed on the ring for
    /// each physical node added later via [`add_node`](Self::add_node). A
    /// higher replica count spreads each physical node's ownership across
    /// more, smaller arcs of the ring, improving distribution evenness at
    /// the cost of more memory and slightly slower `add_node`/`remove_node`
    /// calls (both are `O(replicas * log(total virtual nodes))`). Values in
    /// the 100-200 range are a reasonable default for a handful to a few
    /// dozen physical nodes.
    ///
    /// # Panics
    ///
    /// Panics if `replicas` is `0`. A ring with zero virtual nodes per
    /// physical node can never place any node on the ring, which makes the
    /// ring permanently non-functional — this is a programmer error (a
    /// misconfigured constant), not a runtime condition callers need to
    /// recover from, so it panics immediately at construction rather than
    /// silently building a ring that can never resolve a key.
    ///
    /// ```
    /// use rs_consistenthash::ConsistentHashRing;
    /// let ring = ConsistentHashRing::new(150);
    /// assert!(ring.is_empty());
    /// ```
    pub fn new(replicas: usize) -> Self {
        assert!(
            replicas > 0,
            "ConsistentHashRing::new: `replicas` must be greater than zero"
        );
        Self {
            replicas,
            ring: BTreeMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// Hashes an arbitrary string to a position on the `u64` ring using
    /// [`DefaultHasher`]. `DefaultHasher` is deterministic within a single
    /// build of the standard library (unlike `HashMap`'s randomized
    /// `RandomState`), which is exactly what a stable ring needs: the same
    /// input must always land on the same ring position.
    fn hash_str(value: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Adds a physical node to the ring, placing `replicas` virtual nodes
    /// for it at scattered positions around the ring.
    ///
    /// Adding a node that is already present is a no-op — the call is
    /// idempotent and does not duplicate or move the node's virtual points.
    ///
    /// ```
    /// use rs_consistenthash::ConsistentHashRing;
    /// let mut ring = ConsistentHashRing::new(50);
    /// ring.add_node("cache-1");
    /// assert_eq!(ring.len(), 1);
    /// ring.add_node("cache-1"); // no-op, already present
    /// assert_eq!(ring.len(), 1);
    /// ```
    pub fn add_node(&mut self, name: &str) {
        if self.nodes.contains(name) {
            return;
        }
        self.nodes.insert(name.to_string());
        for i in 0..self.replicas {
            let vnode_id = format!("{name}#{i}");
            let position = Self::hash_str(&vnode_id);
            self.ring.insert(position, name.to_string());
        }
    }

    /// Removes a physical node and all of its virtual nodes from the ring.
    ///
    /// Removing a node that isn't present is a no-op.
    ///
    /// ```
    /// use rs_consistenthash::ConsistentHashRing;
    /// let mut ring = ConsistentHashRing::new(50);
    /// ring.add_node("cache-1");
    /// ring.remove_node("cache-1");
    /// assert!(ring.is_empty());
    /// assert_eq!(ring.get_node("any-key"), None);
    /// ```
    pub fn remove_node(&mut self, name: &str) {
        if !self.nodes.remove(name) {
            return;
        }
        for i in 0..self.replicas {
            let vnode_id = format!("{name}#{i}");
            let position = Self::hash_str(&vnode_id);
            self.ring.remove(&position);
        }
    }

    /// Returns the physical node that owns `key`, or `None` if the ring has
    /// no nodes.
    ///
    /// The key is hashed to a position on the ring, and ownership goes to
    /// the virtual node at the first ring position at-or-after the key's
    /// position, walking clockwise and wrapping back around to the
    /// smallest ring position if the key hashes past every existing point.
    ///
    /// ```
    /// use rs_consistenthash::ConsistentHashRing;
    /// let mut ring = ConsistentHashRing::new(100);
    /// assert_eq!(ring.get_node("k"), None); // empty ring
    ///
    /// ring.add_node("only-node");
    /// assert_eq!(ring.get_node("k"), Some("only-node"));
    /// ```
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let position = Self::hash_str(key);
        match self.ring.range(position..).next() {
            Some((_, name)) => Some(name.as_str()),
            // Wrapped past the largest ring position: clockwise walk
            // continues from the smallest position on the ring.
            None => self.ring.values().next().map(|name| name.as_str()),
        }
    }

    /// Returns up to `n` distinct physical nodes for `key`, in preference
    /// order: the key's owner (exactly what [`get_node`](Self::get_node)
    /// returns) first, then the next distinct nodes walking clockwise around
    /// the ring. This is the "preference list" used to place `n` replicas of
    /// a key, as in Dynamo-style stores.
    ///
    /// Fewer than `n` nodes are returned when the ring has fewer than `n`
    /// physical nodes, and the result is empty for `n == 0` or an empty ring.
    /// Virtual nodes belonging to a node that was already collected are
    /// skipped, so the returned nodes are always distinct.
    ///
    /// ```
    /// use rs_consistenthash::ConsistentHashRing;
    /// let mut ring = ConsistentHashRing::new(100);
    /// for name in ["a", "b", "c", "d"] {
    ///     ring.add_node(name);
    /// }
    /// let replicas = ring.get_nodes("user:42", 3);
    /// assert_eq!(replicas.len(), 3);
    /// assert_eq!(Some(replicas[0]), ring.get_node("user:42"));
    /// ```
    pub fn get_nodes(&self, key: &str, n: usize) -> Vec<&str> {
        let limit = n.min(self.nodes.len());
        let mut found: Vec<&str> = Vec::with_capacity(limit);
        if limit == 0 {
            return found;
        }
        let position = Self::hash_str(key);
        // Clockwise from the key's position, then wrapping to the start.
        let walk = self
            .ring
            .range(position..)
            .chain(self.ring.range(..position));
        for (_, name) in walk {
            let name = name.as_str();
            if !found.contains(&name) {
                found.push(name);
                if found.len() == limit {
                    break;
                }
            }
        }
        found
    }

    /// Returns `true` if a physical node called `name` is in the ring.
    pub fn contains_node(&self, name: &str) -> bool {
        self.nodes.contains(name)
    }

    /// Returns the number of distinct physical nodes currently in the ring.
    ///
    /// This is the count of physical nodes, not virtual nodes — see
    /// [`virtual_node_count`](Self::virtual_node_count) for the latter.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the ring has no physical nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the total number of virtual node points currently placed on
    /// the ring (in the common case, `self.len() * replicas`; it can be
    /// slightly lower if two virtual node identifiers happened to hash to
    /// the same `u64` position, which is exceedingly rare in practice).
    pub fn virtual_node_count(&self) -> usize {
        self.ring.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn empty_ring_returns_none() {
        let ring = ConsistentHashRing::new(10);
        assert_eq!(ring.get_node("anything"), None);
        assert!(ring.is_empty());
    }

    #[test]
    #[should_panic(expected = "replicas` must be greater than zero")]
    fn zero_replicas_panics() {
        let _ = ConsistentHashRing::new(0);
    }

    #[test]
    fn single_node_owns_everything() {
        let mut ring = ConsistentHashRing::new(50);
        ring.add_node("solo");
        for i in 0..500 {
            assert_eq!(ring.get_node(&format!("key-{i}")), Some("solo"));
        }
    }

    #[test]
    fn lookups_are_deterministic_and_stable() {
        let mut ring = ConsistentHashRing::new(100);
        ring.add_node("a");
        ring.add_node("b");
        ring.add_node("c");

        // Repeated lookups of the same key, with no ring changes in
        // between, must always agree.
        for i in 0..1000 {
            let key = format!("stable-key-{i}");
            let first = ring.get_node(&key).map(|s| s.to_string());
            let second = ring.get_node(&key).map(|s| s.to_string());
            assert_eq!(first, second);
        }
    }

    #[test]
    fn add_node_is_idempotent() {
        let mut ring = ConsistentHashRing::new(64);
        ring.add_node("x");
        let vnodes_after_first_add = ring.virtual_node_count();
        ring.add_node("x");
        assert_eq!(ring.virtual_node_count(), vnodes_after_first_add);
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn remove_node_clears_its_virtual_nodes() {
        let mut ring = ConsistentHashRing::new(64);
        ring.add_node("x");
        ring.add_node("y");
        assert_eq!(ring.virtual_node_count(), 128);
        ring.remove_node("x");
        assert_eq!(ring.virtual_node_count(), 64);
        assert_eq!(ring.len(), 1);
        // Removing an absent node is a no-op.
        ring.remove_node("does-not-exist");
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn removed_node_is_never_returned() {
        let mut ring = ConsistentHashRing::new(100);
        for name in ["a", "b", "c", "d"] {
            ring.add_node(name);
        }
        ring.remove_node("c");
        for i in 0..2000 {
            let owner = ring.get_node(&format!("key-{i}")).unwrap();
            assert_ne!(owner, "c");
        }
    }

    /// Property 1: with a reasonable number of virtual nodes per physical
    /// node, keys should be *reasonably* balanced across physical nodes.
    ///
    /// With `replicas = 200` virtual points per physical node, the relative
    /// standard deviation of each node's share of the ring is on the order
    /// of `1/sqrt(200) ~= 7%`, so real observed shares clustering within
    /// +/-40% of the `total_keys / node_count` ideal is an easy, safe bound
    /// (many standard deviations of headroom) while still being a
    /// meaningful assertion that the distribution isn't wildly skewed
    /// (which is what you'd get with only 1 point per node, or a bad hash
    /// function).
    #[test]
    fn keys_are_reasonably_balanced_across_nodes() {
        const NODE_COUNT: usize = 8;
        const REPLICAS: usize = 200;
        const KEY_COUNT: usize = 20_000;
        const TOLERANCE: f64 = 0.40;

        let mut ring = ConsistentHashRing::new(REPLICAS);
        for i in 0..NODE_COUNT {
            ring.add_node(&format!("node-{i}"));
        }

        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..KEY_COUNT {
            let key = format!("balance-key-{i}");
            let owner = ring.get_node(&key).unwrap().to_string();
            *counts.entry(owner).or_insert(0) += 1;
        }

        // Every physical node must have received at least some keys.
        assert_eq!(
            counts.len(),
            NODE_COUNT,
            "expected all {NODE_COUNT} nodes to own at least one key, got {counts:?}"
        );

        let ideal = KEY_COUNT as f64 / NODE_COUNT as f64;
        let lower = ideal * (1.0 - TOLERANCE);
        let upper = ideal * (1.0 + TOLERANCE);

        for (node, count) in &counts {
            let count = *count as f64;
            assert!(
                count >= lower && count <= upper,
                "node {node} owned {count} keys, expected within [{lower}, {upper}] \
                 (ideal {ideal}, tolerance +/-{TOLERANCE})"
            );
        }
    }

    /// Property 2 (THE defining property of consistent hashing): adding a
    /// node to an M-node ring should remap only a small fraction of keys —
    /// close to the theoretical `~1/(M+1)` — instead of remapping nearly
    /// everything, which is what a naive `hash(key) % N` scheme would do
    /// (changing `N` changes the modulus for essentially every key, so a
    /// naive scheme would remap close to 100% of keys here).
    #[test]
    fn adding_a_node_remaps_only_a_small_fraction_of_keys() {
        const INITIAL_NODES: usize = 5;
        const REPLICAS: usize = 150;
        const KEY_COUNT: usize = 10_000;

        let mut ring = ConsistentHashRing::new(REPLICAS);
        for i in 0..INITIAL_NODES {
            ring.add_node(&format!("node-{i}"));
        }

        let keys: Vec<String> = (0..KEY_COUNT).map(|i| format!("remap-key-{i}")).collect();

        let before: Vec<String> = keys
            .iter()
            .map(|k| ring.get_node(k).unwrap().to_string())
            .collect();

        // Grow the ring from M to M+1 nodes.
        ring.add_node(&format!("node-{INITIAL_NODES}"));

        let after: Vec<String> = keys
            .iter()
            .map(|k| ring.get_node(k).unwrap().to_string())
            .collect();

        let remapped = before
            .iter()
            .zip(after.iter())
            .filter(|(b, a)| b != a)
            .count();

        let remapped_fraction = remapped as f64 / KEY_COUNT as f64;
        let ideal_fraction = 1.0 / (INITIAL_NODES + 1) as f64; // ~= 0.1667

        // Every remapped key must have moved specifically to the new node
        // (consistent hashing only ever displaces keys onto the newly
        // inserted node's arcs — it never reshuffles ownership among the
        // pre-existing nodes).
        let new_node_name = format!("node-{INITIAL_NODES}");
        for (b, a) in before.iter().zip(after.iter()) {
            if b != a {
                assert_eq!(
                    a, &new_node_name,
                    "a remapped key moved to {a}, but only the newly added \
                     node {new_node_name} should ever gain keys when it joins"
                );
            }
        }

        // The core assertion: only a small slice of keys moved, within a
        // generous 3x band around the theoretical ~1/(M+1) expectation,
        // and dramatically less than the ~100% a naive `hash(key) % N`
        // modulo scheme would remap when N changes from 5 to 6.
        assert!(
            remapped_fraction > ideal_fraction / 3.0,
            "remapped fraction {remapped_fraction} is implausibly low versus \
             ideal {ideal_fraction} (this would suggest the new node barely \
             received any keys at all)"
        );
        assert!(
            remapped_fraction < ideal_fraction * 3.0,
            "remapped fraction {remapped_fraction} is far above the ideal \
             {ideal_fraction} -- consistent hashing should keep this small"
        );
        assert!(
            remapped_fraction < 0.5,
            "remapped fraction {remapped_fraction} is nowhere near consistent \
             with consistent hashing's promise; a naive modulo scheme would \
             remap close to 100% of keys here, and this implementation should \
             be dramatically better than that"
        );
    }

    /// Sanity check on the wraparound branch specifically: construct a ring
    /// with one node and confirm a key that hashes above every virtual
    /// node's position still resolves (via wraparound to the smallest
    /// position) rather than returning `None`.
    #[test]
    fn wraparound_resolves_keys_past_the_largest_ring_position() {
        let mut ring = ConsistentHashRing::new(100);
        ring.add_node("only");
        // With 100 virtual points and only one physical node, essentially
        // every key must resolve to "only" -- including any key whose hash
        // lands after the ring's single node's largest virtual point,
        // which specifically exercises the wraparound branch.
        for i in 0..5000 {
            assert_eq!(ring.get_node(&format!("wrap-{i}")), Some("only"));
        }
    }

    #[test]
    fn get_nodes_starts_with_the_owner_and_is_distinct() {
        let mut ring = ConsistentHashRing::new(50);
        for name in ["a", "b", "c", "d", "e"] {
            ring.add_node(name);
        }
        for i in 0..500 {
            let key = format!("key-{i}");
            let nodes = ring.get_nodes(&key, 3);
            assert_eq!(nodes.len(), 3, "key {key} should get 3 distinct nodes");
            assert_eq!(Some(nodes[0]), ring.get_node(&key));
            let unique: std::collections::HashSet<_> = nodes.iter().collect();
            assert_eq!(
                unique.len(),
                3,
                "nodes for {key} must be distinct: {nodes:?}"
            );
        }
    }

    #[test]
    fn get_nodes_is_capped_by_the_number_of_physical_nodes() {
        let mut ring = ConsistentHashRing::new(20);
        assert!(ring.get_nodes("k", 3).is_empty(), "empty ring");
        ring.add_node("a");
        ring.add_node("b");
        assert!(ring.get_nodes("k", 0).is_empty(), "n == 0");
        assert_eq!(ring.get_nodes("k", 1).len(), 1);
        let mut all = ring.get_nodes("k", 10);
        all.sort_unstable();
        assert_eq!(all, ["a", "b"]);
    }

    #[test]
    fn get_nodes_next_replica_takes_over_when_the_owner_leaves() {
        // The defining property of a preference list: if the owner is
        // removed, the key's next-listed node becomes its new owner.
        let mut ring = ConsistentHashRing::new(100);
        for name in ["a", "b", "c", "d"] {
            ring.add_node(name);
        }
        for i in 0..300 {
            let key = format!("k{i}");
            let prefs: Vec<String> = ring
                .get_nodes(&key, 2)
                .into_iter()
                .map(String::from)
                .collect();
            let mut without_owner = ConsistentHashRing::new(100);
            for name in ["a", "b", "c", "d"] {
                if name != prefs[0] {
                    without_owner.add_node(name);
                }
            }
            assert_eq!(without_owner.get_node(&key), Some(prefs[1].as_str()));
        }
    }

    #[test]
    fn contains_node_tracks_membership() {
        let mut ring = ConsistentHashRing::new(10);
        assert!(!ring.contains_node("a"));
        ring.add_node("a");
        assert!(ring.contains_node("a"));
        ring.remove_node("a");
        assert!(!ring.contains_node("a"));
    }
}
