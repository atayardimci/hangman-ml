class NGramTrie:
    """Trie data structure to store n-grams of words and their counts."""

    def __init__(self, words: list[str], n: int):
        self.root = {}
        self.end_symbol = "*"
        self.n = n
        self._populate_trie(words)

    def _populate_trie(self, words: list):
        for word in words:
            self._add_ngrams(word)

    def _add_ngrams(self, word: str):
        """Add n-grams of a word to the trie."""
        for i in range(len(word) - self.n + 1):
            ngram = word[i : i + self.n]
            self._insert(ngram)

    def _insert(self, ngram: str):
        """Insert an n-gram into the trie."""
        node = self.root
        for char in ngram:
            if char not in node:
                node[char] = {}
            node = node[char]
        if "count" not in node:
            node["count"] = 0
        node["count"] += 1

    def get_count(self, ngram: str) -> int:
        """Get the count of an n-gram in the trie."""
        node = self.root
        for char in ngram:
            if char not in node:
                return 0
            node = node[char]
        return node.get("count", 0)

    def get_matching_ngrams_with_count(
        self, pattern: str, not_includes: set[str] = set()
    ) -> dict[str, int]:
        """Get all matching n-grams with their counts based on the given pattern."""
        if any(char in not_includes for char in pattern):
            raise ValueError("Invalid not_includes characters in the pattern")
        matches = {}
        self._dfs_match(self.root, pattern, "", not_includes, matches)
        return matches

    def _dfs_match(
        self,
        node: dict,
        pattern: str,
        current_ngram: str,
        not_includes: set,
        matches: dict,
    ):
        """Depth-first search to find matching n-grams."""
        if len(current_ngram) == self.n:
            if "count" in node:
                matches[current_ngram] = node["count"]
        else:
            char = pattern[0]
            if char == "_":
                for child_char, child_node in node.items():
                    if child_char != "count" and child_char not in not_includes:
                        self._dfs_match(
                            child_node,
                            pattern[1:],
                            current_ngram + child_char,
                            not_includes,
                            matches,
                        )
            elif char in node:
                self._dfs_match(
                    node[char], pattern[1:], current_ngram + char, not_includes, matches
                )
