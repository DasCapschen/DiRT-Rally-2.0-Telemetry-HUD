/*
 * Copyright © 2019 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef OVERLAY_PARAMS_H
#define OVERLAY_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define OVERLAY_PARAMS                               \
   OVERLAY_PARAM_BOOL(show_debug)                    \
   OVERLAY_PARAM_BOOL(show_time)                     \
   OVERLAY_PARAM_BOOL(show_progress)                 \
   OVERLAY_PARAM_BOOL(show_gauge)                    \
   OVERLAY_PARAM_BOOL(show_inputs)                   \
   OVERLAY_PARAM_BOOL(show_springs)                  \
   OVERLAY_PARAM_CUSTOM(position)                    \
   OVERLAY_PARAM_CUSTOM(width)                       \
   OVERLAY_PARAM_CUSTOM(height)                      \
   OVERLAY_PARAM_CUSTOM(offset_x)                    \
   OVERLAY_PARAM_CUSTOM(offset_y)                    \
   OVERLAY_PARAM_CUSTOM(fps_limit)                   \
   OVERLAY_PARAM_CUSTOM(vsync)                       \
   OVERLAY_PARAM_CUSTOM(font_size_small)             \
   OVERLAY_PARAM_CUSTOM(font_size_medium)            \
   OVERLAY_PARAM_CUSTOM(font_size_big)               \
   OVERLAY_PARAM_CUSTOM(background_alpha)

enum overlay_param_position {
   LAYER_POSITION_TOP_LEFT,
   LAYER_POSITION_TOP_RIGHT,
   LAYER_POSITION_BOTTOM_LEFT,
   LAYER_POSITION_BOTTOM_RIGHT,
};

enum overlay_param_enabled {
#define OVERLAY_PARAM_BOOL(name) OVERLAY_PARAM_ENABLED_##name,
#define OVERLAY_PARAM_CUSTOM(name)
   OVERLAY_PARAMS
#undef OVERLAY_PARAM_BOOL
#undef OVERLAY_PARAM_CUSTOM
   OVERLAY_PARAM_ENABLED_MAX
};

struct overlay_params {
   bool enabled[OVERLAY_PARAM_ENABLED_MAX];
   enum overlay_param_position position;
   unsigned width;
   unsigned height;
   unsigned offset_x;
   unsigned offset_y;
   uint32_t fps_limit;
   unsigned vsync;
   float font_size_small;
   float font_size_medium;
   float font_size_big;
   float background_alpha;
};

const extern char *overlay_param_names[];

void parse_overlay_env(struct overlay_params *params,
                       const char *env);
void parse_overlay_config(struct overlay_params *params,
                       const char *env);

#ifdef __cplusplus
}
#endif

#endif /* OVERLAY_PARAMS_H */
