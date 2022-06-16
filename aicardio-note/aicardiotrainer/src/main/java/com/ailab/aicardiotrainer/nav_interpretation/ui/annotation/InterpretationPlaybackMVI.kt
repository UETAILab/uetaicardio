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

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import androidx.recyclerview.widget.LinearSmoothScroller
import androidx.recyclerview.widget.RecyclerView
import com.ailab.aicardiotrainer.R
import kotlinx.android.synthetic.main.fragment_annotation.*

class InterpretationPlaybackMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationPlaybackMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationPlaybackMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationPlaybackMVI"

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
//        Log.w(TAG, "renderViewEffect ${viewEffect.javaClass.name}")
        when (viewEffect) {

            is InterpretationViewEffect.RenderMP4Frame -> {
//                Log.w(TAG, "${viewEffect.rmf}")
                renderReadingMediaFrame(annotationFragment, viewEffect)

            }
            is InterpretationViewEffect.RenderComponentActivity -> {
                renderButtonPlayPause(annotationFragment, viewEffect.idComponent, viewEffect.isPlaying)
            }
            is InterpretationViewEffect.RenderTextViewGammaCorrection -> {
                renderTextViewGammaCorrection(annotationFragment, viewEffect)

            }
        }

    }

    private fun renderTextViewGammaCorrection(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect.RenderTextViewGammaCorrection) {
        Log.w(TAG, "renderTextViewGammaCorrection - ${viewEffect.gamma}")
        annotationFragment.tv_gamma_correction.setText(viewEffect.gamma.toString())
    }

    private fun renderButtonPlayPause(annotationFragment: AnnotationFragment, button: Int, isPlaying: Boolean) {
        if (AnnotationFragment.bitmapPlay == null) AnnotationFragment.bitmapPlay = BitmapFactory.decodeResource(annotationFragment.resources, R.drawable.ic_play)
        if (AnnotationFragment.bitmapPause == null) AnnotationFragment.bitmapPause = BitmapFactory.decodeResource(annotationFragment.resources, R.drawable.ic_pause)

        if (button == R.id.bt_play_pause)
            annotationFragment.bt_play_pause.setImageBitmap(if (isPlaying) AnnotationFragment.bitmapPause else AnnotationFragment.bitmapPlay)
    }

    class CenterSmoothScroller(context: Context?) : LinearSmoothScroller(context) {

        override fun calculateDtToFit(viewStart: Int, viewEnd: Int, boxStart: Int, boxEnd: Int, snapPreference: Int): Int {
            return boxStart + (boxEnd - boxStart) / 2 - (viewStart + (viewEnd - viewStart) / 2)
        }

    }

    private fun renderReadingMediaFrame(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect.RenderMP4Frame) {
//        Log.w(TAG, "renderReadingMediaFrame")
        val renderMP4Frame = viewEffect.rmf
        if (renderMP4Frame.numFrame > 0) {
            annotationFragment.inter_iv_draw_canvas.infoText = renderMP4Frame.inforFrame
            annotationFragment.inter_iv_draw_canvas.setCustomImageBitmap(renderMP4Frame.bitmap)
            val frameIndex = renderMP4Frame.idFrame
            if (frameIndex != annotationFragment.interpretationFrameRVAdapter.getCurrentPosition()) {
                val smoothScroller: RecyclerView.SmoothScroller = CenterSmoothScroller(annotationFragment.inter_rv_frames.getContext())
                smoothScroller.setTargetPosition(frameIndex)
                annotationFragment.inter_rv_frames.layoutManager?.startSmoothScroll(smoothScroller)
                annotationFragment.interpretationFrameRVAdapter.setCurrentPosition(frameIdx=frameIndex)
            }

        }

    }

    private fun renderViewState(annotationFragment: AnnotationFragment, viewState: InterpretationViewState) {
    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        interpretationActVM.viewStates().value?.let {
            interpretationActVM.reduce(PlaybackReducer(interpretationActVM, it, interpretationViewEvent))
        }

    }

    inner class PlaybackReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        private val numFrame get() = viewModel.numFrame
        private val isPlaying get() = viewModel.getIsPlaying()
        private val currentFrameIndex get() = viewModel.getCurrentFrameIndex()

        override fun reduce(): StateEffectObject {
            when(viewEvent) {

                InterpretationViewEvent.NextFrame -> {
                    return playNextFrame()
                }

                InterpretationViewEvent.ShowNextFrame -> {
                    Log.w(TAG, "InterpretationViewEvent.ShowNextFrame")
                    return showNextFrame()
                }

                InterpretationViewEvent.ShowPreviousFrame -> {
                    Log.w(TAG, "InterpretationViewEvent.ShowPreviousFrame")
                    return showPreviousFrame()
                }

                InterpretationViewEvent.ShowFirstFrame -> {
                    Log.w(TAG, "InterpretationViewEvent.ShowFirstFrame")
                    return showFirstFrame()
                }

                InterpretationViewEvent.ShowLastFrame -> {
                    Log.w(TAG, "InterpretationViewEvent.ShowLastFrame")
                    return showLastFrame()
                }
                InterpretationViewEvent.PlayPauseVideo -> {
                    playPauseVideo()
                    return StateEffectObject(
                        viewEffect = InterpretationViewEffect.RenderComponentActivity(idComponent = R.id.bt_play_pause, isPlaying = viewModel.getIsPlaying())
                    )
                }


//                is InterpretationViewEvent.ShowEsvEdvOrReadingMediaFrame -> {
//                    Log.w(TAG, "ReadingMediaViewEvent.ShowEsvEdvOrReadingMediaFrame ${viewEvent.isEsvEdv}")
//                    return showEsvEdvOrReadingMediaFrame(viewEvent.isEsvEdv)
//                }

            }
            return StateEffectObject()
        }
        private fun playNextFrame(): StateEffectObject {
            if (numFrame > 0 && isPlaying == true) {

                val nextFrame = (currentFrameIndex + 1) % numFrame

                viewModel.setCurrentFrameIndex(nextFrame)
                return StateEffectObject(null, viewModel.getRenderReadingMediaFrame())
            }
            return StateEffectObject()

        }

        private fun showFirstFrame(): StateEffectObject {
            if (numFrame > 0) {
                viewModel.setCurrentFrameIndex(0)
                viewModel.setIsPlaying(isPlayingValue = false)
                return StateEffectObject(null, viewModel.getRenderReadingMediaFrame())
            }
            return StateEffectObject()
        }

        private fun showLastFrame() : StateEffectObject{
            if (numFrame > 0) {
                viewModel.setCurrentFrameIndex(numFrame - 1)
                viewModel.setIsPlaying(isPlayingValue = false)
                return StateEffectObject(null, viewModel.getRenderReadingMediaFrame())
            }
            return StateEffectObject()
        }

        private fun showNextFrame(): StateEffectObject {
            if (numFrame > 0) {
                viewModel.setIsPlaying(isPlayingValue = false)
                viewModel.setCurrentFrameIndex((currentFrameIndex + 1) % numFrame)
//                Log.w(TAG, "showNextFrame $isPlaying $currentFrameIndex")
                return StateEffectObject(null, viewModel.getRenderReadingMediaFrame())
            }
            return StateEffectObject()
        }

        private fun showPreviousFrame(): StateEffectObject {
            if (numFrame > 0) {
                viewModel.setIsPlaying(isPlayingValue = false)
                val frameIdx = if (currentFrameIndex <= 0) numFrame - 1 else currentFrameIndex - 1
                viewModel.setCurrentFrameIndex(frameIdx)
                return StateEffectObject(null, viewModel.getRenderReadingMediaFrame())
            }
            return StateEffectObject()
        }

        private fun playPauseVideo() {
            viewModel.setIsPlaying(isPlayingValue = !isPlaying)

        }

    }

    inner class ReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            return StateEffectObject()
        }

    }


}