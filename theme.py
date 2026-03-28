"""
theme.py — "Soft Professional Tech" Gradio theme.

Extends Gradio's built-in Soft base with a teal-slate primary palette,
muted cancel colours, and Inter / JetBrains Mono type pairing.
"""

from typing import Iterable

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# ---------------------------------------------------------------------------
# Custom colour ramps
# ---------------------------------------------------------------------------

colors.teal_slate = colors.Color(
    name="teal_slate",
    c50="#eef4f7",
    c100="#d5e6ee",
    c200="#aecde0",
    c300="#80b1ce",
    c400="#5595bc",
    c500="#3a7a9f",
    c600="#306685",
    c700="#28536c",
    c800="#204358",
    c900="#183447",
    c950="#102738",
)

colors.muted_red = colors.Color(
    name="muted_red",
    c50="#fdf3f3",
    c100="#fce5e5",
    c200="#f8c0c0",
    c300="#f39a9a",
    c400="#ec7373",
    c500="#e05858",
    c600="#c94747",
    c700="#a33838",
    c800="#862f2f",
    c900="#6e2727",
    c950="#571f1f",
)


# ---------------------------------------------------------------------------
# Theme class
# ---------------------------------------------------------------------------

class SoftProTheme(Soft):
    """
    "Soft Professional Tech" — balances enterprise rigidity with
    consumer-grade approachability.  Builds on Gradio's Soft base.
    """

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.slate,
        secondary_hue: colors.Color | str = colors.teal_slate,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # Surfaces
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(145deg, *primary_200 0%, *primary_100 60%, *secondary_50 100%)",
            body_background_fill_dark="linear-gradient(145deg, *primary_950 0%, *primary_900 60%, *secondary_950 100%)",

            # Primary buttons (Send / submit)
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_400, *secondary_500)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",

            # Secondary buttons (Clear / auxiliary)
            button_secondary_text_color="*secondary_700",
            button_secondary_text_color_hover="*secondary_800",
            button_secondary_background_fill="linear-gradient(90deg, *primary_100, *primary_200)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_200, *primary_300)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_700, *primary_800)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_600, *primary_700)",

            # Cancel / destructive buttons
            button_cancel_background_fill=f"linear-gradient(90deg, {colors.muted_red.c400}, {colors.muted_red.c500})",
            button_cancel_background_fill_dark=f"linear-gradient(90deg, {colors.muted_red.c700}, {colors.muted_red.c800})",
            button_cancel_background_fill_hover=f"linear-gradient(90deg, {colors.muted_red.c500}, {colors.muted_red.c600})",
            button_cancel_background_fill_hover_dark=f"linear-gradient(90deg, {colors.muted_red.c800}, {colors.muted_red.c900})",
            button_cancel_text_color="white",
            button_cancel_text_color_dark="white",
            button_cancel_text_color_hover="white",
            button_cancel_text_color_hover_dark="white",

            # Accent & block chrome
            slider_color="*secondary_400",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="2px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_100",
        )


theme = SoftProTheme()
