/*
 * Copyright 2021 UET-AILAB
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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.graphics.Bitmap
import android.view.MotionEvent

class StudyGVItem (
    val id: Long,
    val name: String,
    val img_path: String
)

data class FragmentViewObject(
    val numFrame: Int,
    val frameID: Int,
    val frameText: String,
    val bitmap: Bitmap?,
    val isPlaying: Boolean,
    val isESV: Boolean,
    val isEDV: Boolean
)

data class FrameCanvasItem(
    val index: Int,
    val bitmap: Bitmap
)

data class TouchEventObject(
    val frameIdx: Int,
    val scale: Float,
    val key: String = "",
    val event: MotionEvent?,
    val ix: Float,
    val iy: Float,
    val isLongClicked: Boolean = false,
    val modifyPointIdx: Int = -1,
    val modifyBoundaryIndex : Pair<Int, Int> = Pair(-1, -1)
)

data class EFObject(
    val indexFrame: Int = -1,
    val indexESV: Int = -1,
    val indexEDV: Int = -1,
    val volumeESV: Float = 0F,
    val volumeEDV: Float = 0F,
    val efValue: Float = -1.0F
)
