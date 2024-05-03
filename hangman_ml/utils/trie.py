class Trie:
    """Trie data structure to store words and search for matches."""

    def __init__(self, words: list[str]):
        self.root = {}
        self.end_symbol = "*"
        self._populate_trie(words)

    def _populate_trie(self, words: list):
        """Populate the trie with the given words."""
        for word in words:
            self.insert(word)

    def insert(self, word: str):
        """Insert a word into the trie."""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_symbol] = True

    def get_matches(self, masked_word: str, not_includes: set[str] = set()):
        """Get all words in the trie that match the given masked_word.
        The word can contain the wildcard character `'_'`.

        If not_includes is provided, the matches will be filtered
        """
        matches = []
        self._dfs_search(self.root, masked_word, not_includes, "", matches)
        return matches

    def _dfs_search(
        self,
        node: dict,
        suffix: str,
        not_includes: set[str],
        current_word: str,
        matches: list,
    ):
        """Depth-first search to find all words that match the given suffix."""
        if not suffix:
            if self.end_symbol in node:
                matches.append(current_word)
        else:
            char = suffix[0]
            if char == "_":
                for child_char, child_node in node.items():
                    if child_char != self.end_symbol and child_char not in not_includes:
                        self._dfs_search(
                            child_node,
                            suffix[1:],
                            not_includes,
                            current_word + child_char,
                            matches,
                        )
            elif char in node:
                self._dfs_search(
                    node[char], suffix[1:], not_includes, current_word + char, matches
                )
