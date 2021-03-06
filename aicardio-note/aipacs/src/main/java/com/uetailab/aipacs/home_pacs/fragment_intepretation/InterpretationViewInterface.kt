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

import android.graphics.Canvas
import android.view.MotionEvent
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewVM

interface OnStudyPreviewClicked {
    fun onStudyPreviewClicked(item: StudyGVItem)
    fun onStudyPreviewLongClicked(item: StudyGVItem): Boolean
}
interface OnDrawListener {
    fun draw(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?)
}

interface OnNormalizeTouchListener {
    fun onTouchEvent(view: InterpretationViewStudyPreviewCanvasView, event: MotionEvent?, ix: Float, iy: Float)
}

interface OnHomeViewDataPass {
    fun onHomeViewViewModelPass(viewModel: HomeViewVM)
}