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

package com.ailab.aicardiotrainer.interpretation

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
import kotlinx.android.synthetic.main.activity_interpretation.*

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

        fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationActivity, viewState)
        }

        fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationActivity, viewEffect)
        }
    }

    private fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
        when (viewEffect) {

            is InterpretationViewEffect.RenderMP4Frame -> {
//                Log.w(TAG, "${viewEffect.rmf}")
                renderReadingMediaFrame(interpretationActivity, viewEffect)

            }
            is InterpretationViewEffect.RenderComponentActivity -> {
                renderButtonPlayPause(interpretationActivity, viewEffect.idComponent, viewEffect.isPlaying)
            }
            is InterpretationViewEffect.RenderTextViewGammaCorrection -> {
                renderTextViewGammaCorrection(interpretationActivity, viewEffect)

            }
        }

    }

    private fun renderTextViewGammaCorrection(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect.RenderTextViewGammaCorrection) {
        interpretationActivity.tv_gamma_correction.setText(viewEffect.gamma.toString())
    }

    private fun renderButtonPlayPause(interpretationActivity: InterpretationActivity, button: Int, isPlaying: Boolean) {
        if (InterpretationActivity.bitmapPlay == null) InterpretationActivity.bitmapPlay = BitmapFactory.decodeResource(interpretationActivity.resources, R.drawable.ic_play)
        if (InterpretationActivity.bitmapPause == null) InterpretationActivity.bitmapPause = BitmapFactory.decodeResource(interpretationActivity.resources, R.drawable.ic_pause)

        if (button == R.id.bt_play_pause)
            interpretationActivity.bt_play_pause.setImageBitmap(if (isPlaying) InterpretationActivity.bitmapPause else InterpretationActivity.bitmapPlay)
    }

    class CenterSmoothScroller(context: Context?) : LinearSmoothScroller(context) {

        override fun calculateDtToFit(viewStart: Int, viewEnd: Int, boxStart: Int, boxEnd: Int, snapPreference: Int): Int {
            return boxStart + (boxEnd - boxStart) / 2 - (viewStart + (viewEnd - viewStart) / 2)
        }

    }

    private fun renderReadingMediaFrame(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect.RenderMP4Frame) {
//        Log.w(TAG, "renderReadingMediaFrame")
        val renderMP4Frame = viewEffect.rmf
        if (renderMP4Frame.numFrame > 0) {

            interpretationActivity.inter_iv_draw_canvas.infoText = renderMP4Frame.inforFrame
            interpretationActivity.inter_iv_draw_canvas.setCustomImageBitmap(renderMP4Frame.bitmap)

            val frameIndex = renderMP4Frame.idFrame
            if (frameIndex != interpretationActivity.interpretationFrameRVAdapter.getCurrentPosition()) {
                val smoothScroller: RecyclerView.SmoothScroller = CenterSmoothScroller(interpretationActivity.inter_rv_frames.getContext())
                smoothScroller.setTargetPosition(frameIndex)
                interpretationActivity.inter_rv_frames.layoutManager?.startSmoothScroll(smoothScroller)
                interpretationActivity.interpretationFrameRVAdapter.setCurrentPosition(frameIdx=frameIndex)
            }

            interpretationActivity.cb_esv.isChecked = renderMP4Frame.isESV
            interpretationActivity.cb_edv.isChecked = renderMP4Frame.isEDV
        }

    }

    private fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
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