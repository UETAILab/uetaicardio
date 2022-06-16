/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardiotrainer.repositories

import android.graphics.Bitmap

class RenderAnnotation(
    val isPlaying: Boolean,
    val numFrame: Int,
    val indexFrame: Int,
    val infoText: String,
    val esvEdvText: String,
    val bitmap: Bitmap,
    val isESV: Boolean,
    val isEDV: Boolean,
    val length: Float = 0F,
    val area: Float = 0F,
    val volume: Float = 0F,
    val nPointGLS: Int = 0,
    val nPointsEF: Int = 0,
    val ef: EFObject = EFObject()
)
