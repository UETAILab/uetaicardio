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

package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

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

import android.graphics.Bitmap
import android.util.Log
import org.opencv.core.Mat

class InterpretationEnhanceBitmapMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationEnhanceBitmapMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationEnhanceBitmapMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationEnhanceBitmapMVI"

        fun process(interpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationActVM, InterpretationViewEvent)
        }

        fun renderViewState(annotationFragment: AnnotationFragment, viewState: InterpretationViewState) {
            getInstance().renderViewState(annotationFragment, viewState)
        }

        fun renderViewEffect(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(annotationFragment, viewEffect)
        }
    }

    private fun renderViewEffect(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect) {
    }

    private fun renderViewState(annotationFragment: AnnotationFragment, viewState: InterpretationViewState) {
        when (viewState.status) {
//            InterpretationViewStatus.EnhanceContrastBitmap -> {
//                Log.w(TAG, "EnhanceContrastBitmap" )
//                if (interpretationActivity.viewModel.getIsPlaying() == false) {
//                    // video is pause then can enhace image
//                    interpretationActivity.viewModel.getRenderReadingMediaFrame()?.let {
//                        InterpretationPlaybackMVI.renderViewEffect(interpretationActivity, InterpretationViewEffect.RenderMP4Frame((it as InterpretationViewEffect.RenderMP4Frame).rmf ))
//                    }
//
//                }
//
//            }
        }
    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.EnhanceContrastBitmap -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(EnhanceContrastBitmapReducer(
                        interpretationActVM, it, interpretationViewEvent
                    ))
                }
            }

        }
    }

    inner class EnhanceContrastBitmapReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.EnhanceContrastBitmap)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StateEffectObject {
            val bitmap = viewModel.getCurrentFrameBitmap()
            val frameIndex = viewModel.getCurrentFrameIndex()

//            val mat = bitmap.toMat()

            val mat = Mat()

//            val newBitmap = mat.canny(bitmap = bitmap, threshold1 = viewEvent.threshold.toDouble())
            val threshold = viewEvent.threshold.toDouble()

            var gamma = threshold / 85.0

            if (gamma > 0) gamma = gamma else gamma = 0.1
            val newBitmap = bitmap.adjustGammaCorrection(gamma = gamma)

//
//            val sopInstanceUIDBitmaps = viewState.sopInstanceUIDBitmaps
//            val newSopInstanceUIDBitmaps = ArrayList<Bitmap> ()
//
//            // sopInstanceUIDBitmaps
//            sopInstanceUIDBitmaps.forEachIndexed { index, bitmap ->
//                if (index == frameIndex) {
//                    newSopInstanceUIDBitmaps.add(newBitmap)
//                } else {
//                    newSopInstanceUIDBitmaps.add(bitmap)
//
//                }
//            }
//
            val rmf = viewModel.getRenderMP4FrameObject()

            rmf?.let {
                return StateEffectObject(
                    viewEffect = InterpretationViewEffect.RenderMP4Frame(it.copy(bitmap = newBitmap))
                )
            } ?: kotlin.run {
                return StateEffectObject()
            }


        }

    }

    inner class ReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, viewEvent: InterpretationViewEvent) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            return StateEffectObject()
        }

    }
}