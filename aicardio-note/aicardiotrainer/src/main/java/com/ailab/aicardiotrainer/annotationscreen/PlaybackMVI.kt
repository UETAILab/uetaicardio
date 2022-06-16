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

package com.ailab.aicardiotrainer.annotationscreen

import android.graphics.BitmapFactory
import android.util.Log
import androidx.recyclerview.widget.RecyclerView
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.repositories.RenderAnnotation
import kotlinx.android.synthetic.main.activity_annotation.*
import kotlinx.android.synthetic.main.layout_calculator.view.*

class PlaybackMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: PlaybackMVI? = null

        const val TAG = "PlaybackUniDirectionMVI"
        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: PlaybackMVI()
                        .also { instance = it }
            }

        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance().process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
            getInstance().renderViewState(annotationActivity, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance().renderViewEffect(annotationActivity, viewEffect)
        }

    }

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
        when (viewEffect) {
            is AnnotationViewEffect.RenderButtonPlayPause -> {
                renderButtonPlayPause(annotationActivity, viewEffect.button, viewEffect.isPlaying)
            }
            is AnnotationViewEffect.RenderAnnotationFrame -> {
                renderAnnotationFrame(annotationActivity, viewEffect.renderAnnotation)
            }
        }
    }

    fun renderAnnotationFrame(annotationActivity: AnnotationActivity, renderAnnotation: RenderAnnotation) {
        Log.w(AnnotationActivity.TAG, "renderAnnotationFrame")
        if (renderAnnotation.numFrame > 0) {
            annotationActivity.iv_draw_canvas.infoText = renderAnnotation.infoText

            annotationActivity.cb_esv.isChecked = renderAnnotation.isESV
            annotationActivity.cb_edv.isChecked = renderAnnotation.isEDV
            annotationActivity.iv_draw_canvas.infoPointEF = "EF: ${renderAnnotation.nPointsEF}"
            annotationActivity.iv_draw_canvas.infoPointGLS = "GLS: ${renderAnnotation.nPointGLS}"
            annotationActivity.iv_draw_canvas.infoEsvEdv = renderAnnotation.esvEdvText
            annotationActivity.iv_draw_canvas.setCustomImageBitmap(renderAnnotation.bitmap)

//            rv_frame_info_list.smoothScrollToPosition(renderAnnotation.indexFrame)
            if (renderAnnotation.indexFrame != annotationActivity.newsRvFrameAdapter.getCurrentPosition()) {
                val smoothScroller: RecyclerView.SmoothScroller = AnnotationActivity.CenterSmoothScroller(annotationActivity.rv_frame_info_list.getContext())
                smoothScroller.setTargetPosition(renderAnnotation.indexFrame)
                annotationActivity.rv_frame_info_list.layoutManager?.startSmoothScroll(smoothScroller)
                annotationActivity.newsRvFrameAdapter.setCurrentPosition(frameIdx=renderAnnotation.indexFrame)
            }


            val ef = renderAnnotation.ef
            annotationActivity.layout_calculator.tv_label_length.text = "Length: %.2f mm".format(renderAnnotation.length)
            annotationActivity.layout_calculator.tv_label_area.text = "Area: %.2f cm2".format(renderAnnotation.area / 100.0F )
            annotationActivity.layout_calculator.tv_label_volume.text = "Volume: %.1f mL".format(renderAnnotation.volume)

            annotationActivity.layout_calculator.tv_label_ESV.text = "LV ESV: %.2f mL ID: %d".format(ef.volumeESV, ef.indexESV)
            annotationActivity.layout_calculator.tv_label_EDV.text = "LV EDV: %.2f mL ID: %d".format(ef.volumeEDV, ef.indexEDV)

            val ef_LV = kotlin.math.round(ef.efValue * 10000) / 100
            annotationActivity.layout_calculator.tv_label_EF.text = "EF: %.2f".format(ef_LV)

        }
    }

    private fun renderButtonPlayPause(annotationActivity: AnnotationActivity, button: Int, isPlaying: Boolean) {
        if (AnnotationActivity.bitmapPlay == null) AnnotationActivity.bitmapPlay = BitmapFactory.decodeResource(annotationActivity.resources, R.drawable.ic_play)
        if (AnnotationActivity.bitmapPause == null) AnnotationActivity.bitmapPause = BitmapFactory.decodeResource(annotationActivity.resources, R.drawable.ic_pause)

        if (button == R.id.bt_play_pause)
            annotationActivity.bt_play_pause.setImageBitmap(if (isPlaying) AnnotationActivity.bitmapPause else AnnotationActivity.bitmapPlay)
    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {

    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {

        annotationActVM.viewStates().value?.let {
            annotationActVM.reduce(PlaybackReducer(annotationActVM, it, annotationViewEvent))
        }

    }

    inner class PlaybackReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        private val numFrame get() = viewModel.numFrame
        private val isPlaying get() = viewModel.getIsPlaying()
        private val currentFrameIndex get() = viewModel.getCurrentFrameIndex()

        override fun reduce(): AnnotationStateEffectObject {
            when(viewEvent) {

                is AnnotationViewEvent.PlayPauseVideo -> {
                    playPauseVideo()
                    return AnnotationStateEffectObject(null, AnnotationViewEffect.RenderButtonPlayPause(R.id.bt_play_pause, isPlaying))
                }

                is AnnotationViewEvent.NextFrame -> {
                    return playNextFrame()
                }

                is AnnotationViewEvent.ShowNextFrame -> {
                    Log.w(TAG, "AnnotationViewEvent.ShowNextFrame")
                    return showNextFrame()
                }

                is AnnotationViewEvent.ShowPreviousFrame -> {
                    Log.w(TAG, "AnnotationViewEvent.ShowPreviousFrame")
                    return showPreviousFrame()
                }

                is AnnotationViewEvent.ShowFirstFrame -> {
                    Log.w(TAG, "AnnotationViewEvent.ShowFirstFrame")
                    return showFirstFrame()
                }

                is AnnotationViewEvent.ShowLastFrame -> {
                    Log.w(TAG, "AnnotationViewEvent.ShowLastFrame")
                    return showLastFrame()
                }

                is AnnotationViewEvent.ShowEsvEdvOrAnnotationFrame -> {
                    Log.w(TAG, "AnnotationViewEvent.ShowEsvEdvOrAnnotationFrame")
                    return showEsvEdvOrAnnotationFrame(viewEvent.isEsvEdv)
                }
                else -> return AnnotationStateEffectObject()
            }

        }

        private fun playNextFrame(): AnnotationStateEffectObject {
            if (numFrame > 0 && isPlaying == true) {
                val nextFrame = (currentFrameIndex + 1) % numFrame
                viewModel.setCurrentFrameIndex(nextFrame)
                return AnnotationStateEffectObject(null, viewModel.getRenderAnnotationFrame())
            }
            return AnnotationStateEffectObject()

        }

        private fun showFirstFrame(): AnnotationStateEffectObject {
            if (numFrame > 0) {
                viewModel.setCurrentFrameIndex(0)
                viewModel.setIsPlaying(isPlayingValue = false)
                return AnnotationStateEffectObject(null, viewModel.getRenderAnnotationFrame())
            }
            return AnnotationStateEffectObject()
        }

        private fun showLastFrame() : AnnotationStateEffectObject{
            if (numFrame > 0) {
                viewModel.setCurrentFrameIndex(numFrame - 1)
                viewModel.setIsPlaying(isPlayingValue = false)
                return AnnotationStateEffectObject(null, viewModel.getRenderAnnotationFrame())
            }
            return AnnotationStateEffectObject()
        }

        private fun showNextFrame(): AnnotationStateEffectObject {
            if (numFrame > 0) {
                viewModel.setIsPlaying(isPlayingValue = false)
                viewModel.setCurrentFrameIndex((currentFrameIndex + 1) % numFrame)
//                Log.w(TAG, "showNextFrame $isPlaying $currentFrameIndex")
                return AnnotationStateEffectObject(null, viewModel.getRenderAnnotationFrame())
            }
            return AnnotationStateEffectObject()
        }

        private fun showPreviousFrame(): AnnotationStateEffectObject {
            if (numFrame > 0) {
                viewModel.setIsPlaying(isPlayingValue = false)
                val frameIdx = if (currentFrameIndex <= 0) numFrame - 1 else currentFrameIndex - 1
                viewModel.setCurrentFrameIndex(frameIdx)
                return AnnotationStateEffectObject(null, viewModel.getRenderAnnotationFrame())
            }
            return AnnotationStateEffectObject()
        }

        private fun playPauseVideo() {
            viewModel.setIsPlaying(isPlayingValue = !isPlaying)
//            Log.w(TAG, "playPauseVideo $isPlaying")
        }

        private fun showEsvEdvOrAnnotationFrame(isEsvEdv: Boolean): AnnotationStateEffectObject {
            if (numFrame > 0) {
                viewModel.setIsPlaying(isPlayingValue = false)
                val frameIdx: Int
                if (isEsvEdv) frameIdx = viewState.dicomAnnotation.getNextEsvEdvFrameIndex(nFrame=numFrame, frameIdx = currentFrameIndex)
                else frameIdx = viewState.dicomAnnotation.getNextAnnoatationFrameIndex(nFrame=numFrame, frameIdx = currentFrameIndex)
                viewModel.setCurrentFrameIndex(frameIdx)
            }
            return AnnotationStateEffectObject()
        }
    }

}