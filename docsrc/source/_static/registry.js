// Filter the registry cards as the reader types. Everything a card can be
// searched by is on its data-search attribute, so this reads no DOM text.
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".rc-block").forEach(function (block) {
        var input = block.querySelector(".rc-search");
        var count = block.querySelector(".rc-count span");
        var cards = Array.prototype.slice.call(block.querySelectorAll(".rc-card"));
        if (!input) {
            return;
        }
        input.addEventListener("input", function () {
            var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
            var shown = 0;
            cards.forEach(function (card) {
                var haystack = card.dataset.search || "";
                var match = terms.every(function (term) {
                    return haystack.indexOf(term) !== -1;
                });
                card.hidden = !match;
                if (match) {
                    shown += 1;
                }
            });
            if (count) {
                count.textContent = String(shown);
            }
        });
    });
});
