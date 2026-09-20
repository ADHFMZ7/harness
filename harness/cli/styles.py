# styles.py
# colour schemes for the terminal front-end

from dataclasses import dataclass, field

from rich.theme import Theme

'''
The front-end never names a colour. It uses these styles, and a scheme says
what each one looks like:

    harness.accent     the prompt marker and the name in the banner
    harness.muted      secondary text: thought times, tool arguments and results, hints
    harness.thinking   the model's reasoning while it streams
    harness.tool       the ⚒ before a tool call
    harness.tool.name  the tool's name
    harness.error      a tool that failed, or a turn that did

Answers are markdown, so a scheme also sets rich's own markdown.* styles and
the pygments theme for code blocks.
'''


@dataclass(frozen=True)
class Scheme:
    name:       str
    summary:    str
    code_theme: str
    styles:     dict[str, str] = field(default_factory=dict)

    @property
    def theme(self) -> Theme:
        return Theme(self.styles)


GRAPHITE = Scheme(
    "graphite", "one accent, structure in grey", "github-dark",
    {
        "harness.accent":        "bold #8fb3d9",
        "harness.muted":         "#6b7079",
        "harness.thinking":      "italic #6b7079",
        "harness.tool":          "#6b7079",
        "harness.tool.name":     "bold #8fb3d9",
        "harness.error":         "#d98c8c",

        "markdown.h1":           "bold #e8eaed",
        "markdown.h2":           "bold #8fb3d9",
        "markdown.h3":           "bold #c8ccd2",
        "markdown.strong":       "bold #eceef1",
        "markdown.code":         "#cfd5dd on #30343b",
        "markdown.block_quote":  "italic #9aa0a9",
        "markdown.list":         "#6b7079",
        "markdown.item.bullet":  "#6b7079",
        "markdown.item.number":  "#6b7079",
        "markdown.hr":           "#3b3f46",
        "markdown.link":         "#8fb3d9",
        "markdown.link_url":     "underline #8fb3d9",
        "markdown.table.border": "#41454d",
        "markdown.table.header": "bold #e8eaed",
    },
)

TIDE = Scheme(
    "tide", "cool, from Nord", "nord",
    {
        "harness.accent":        "bold #88c0d0",
        "harness.muted":         "#616e88",
        "harness.thinking":      "italic #616e88",
        "harness.tool":          "#81a1c1",
        "harness.tool.name":     "bold #81a1c1",
        "harness.error":         "#bf616a",

        "markdown.h1":           "bold #88c0d0",
        "markdown.h2":           "bold #88c0d0",
        "markdown.h3":           "bold #8fbcbb",
        "markdown.strong":       "bold #eceff4",
        "markdown.code":         "#a3be8c",
        "markdown.block_quote":  "italic #81a1c1",
        "markdown.list":         "#5e81ac",
        "markdown.item.bullet":  "#5e81ac",
        "markdown.item.number":  "#5e81ac",
        "markdown.hr":           "#4c566a",
        "markdown.link":         "#88c0d0",
        "markdown.link_url":     "underline #88c0d0",
        "markdown.table.border": "#4c566a",
        "markdown.table.header": "bold #d8dee9",
    },
)

EMBER = Scheme(
    "ember", "warm, from Gruvbox", "gruvbox-dark",
    {
        "harness.accent":        "bold #fabd2f",
        "harness.muted":         "#928374",
        "harness.thinking":      "italic #928374",
        "harness.tool":          "#d65d0e",
        "harness.tool.name":     "bold #fe8019",
        "harness.error":         "#fb4934",

        "markdown.h1":           "bold #fabd2f",
        "markdown.h2":           "bold #fabd2f",
        "markdown.h3":           "bold #83a598",
        "markdown.strong":       "bold #fbf1c7",
        "markdown.code":         "#8ec07c",
        "markdown.block_quote":  "italic #a89984",
        "markdown.list":         "#d65d0e",
        "markdown.item.bullet":  "#d65d0e",
        "markdown.item.number":  "#d65d0e",
        "markdown.hr":           "#504945",
        "markdown.link":         "#83a598",
        "markdown.link_url":     "underline #83a598",
        "markdown.table.border": "#504945",
        "markdown.table.header": "bold #ebdbb2",
    },
)

# Rich's own markdown styles and the terminal's ANSI colours, as it used to be.
CLASSIC = Scheme(
    "classic", "rich's defaults, in your terminal's colours", "monokai",
    {
        "harness.accent":        "bold cyan",
        "harness.muted":         "dim",
        "harness.thinking":      "dim italic",
        "harness.tool":          "magenta",
        "harness.tool.name":     "bold magenta",
        "harness.error":         "red",
    },
)

# In the order /style cycles through them; the first is the default.
SCHEMES = {scheme.name: scheme for scheme in (GRAPHITE, TIDE, EMBER, CLASSIC)}
DEFAULT = GRAPHITE
