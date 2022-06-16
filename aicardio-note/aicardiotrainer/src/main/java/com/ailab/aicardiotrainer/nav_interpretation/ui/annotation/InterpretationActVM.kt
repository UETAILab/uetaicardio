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

import aacmvi.AacMviViewModel
import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import java.lang.Math.round
import kotlin.math.max

class InterpretationActVM(applicationAnnotate: Application) : AacMviViewModel<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent>(applicationAnnotate) {

    companion object {
        const val TAG = "InterpretationActVM"
    }

    init {
        viewState = InterpretationViewState(
            status = InterpretationViewStatus.Start,
            studyInstanceUIDPath = "",
            studyHashMap = HashMap(),
            studyFiles = emptyList(),
            studyRepresentation = emptyList(),
            sopInstanceUIDPath = "",
            sopInstanceUIDBitmaps = emptyList(),
            sopInstanceUIDTags = JSONObject()
        )
    }

    val numFrame get() =  viewState.sopInstanceUIDBitmaps.size

    private var currentFrameIndex: Int = 0
    private var isPlaying: Boolean = true
    private var gammaValue = 1.0

    override fun process(viewEvent: InterpretationViewEvent) {
        super.process(viewEvent)
        when(viewEvent) {
            is InterpretationViewEvent.SopInstanceUIDFrameClicked -> SopInstanceUIDFrameClicked(viewEvent)
            is InterpretationViewEvent.OnChangeGammaCorrection -> OnChangeGammaCorrection(viewEvent)
        }

    }
    private fun SopInstanceUIDFrameClicked(viewEvent: InterpretationViewEvent.SopInstanceUIDFrameClicked) {
        currentFrameIndex = viewEvent.frameItem.index
        getRenderReadingMediaFrame()?.let { viewEffect = it }
    }

    fun reduce(reducer: InterpretationActReducer) {
        val result = reducer.reduce()
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }

    fun reduce(stateEffectObject: StateEffectObject) {
        stateEffectObject.viewState?.let { viewState = it }
        stateEffectObject.viewEffect?.let { viewEffect = it }
    }

    fun getIsPlaying(): Boolean {
        return isPlaying
    }

    fun getCurrentFrameIndex(): Int {
        return currentFrameIndex
    }

    fun setCurrentFrameIndex(nextFrame: Int) {
        currentFrameIndex = nextFrame
    }

    fun setIsPlaying(isPlayingValue: Boolean) {
        isPlaying = isPlayingValue
    }

    fun Double.round(decimals: Int): Double {
        var multiplier = 1.0
        repeat(decimals) { multiplier *= 10 }
        return round(this * multiplier) / multiplier
    }

    fun OnChangeGammaCorrection(viewEvent: InterpretationViewEvent.OnChangeGammaCorrection) {
//        gammaValue = (max(0.1, (viewEvent.threshold) / 85.0) ) as Double) .round(1)
        gammaValue = round((max(0.1, (viewEvent.threshold) / 85.0)) * 10).toDouble() / 10
        Log.w(TAG, "OnChangeGammaCorrection: ${gammaValue}")
        getRenderReadingMediaFrame()?.let {
            Log.w(TAG, "getRenderReadingMediaFrame: ${gammaValue}")

            viewEffect = it
        }

    }
    fun getRenderReadingMediaFrame(): InterpretationViewEffect.RenderMP4Frame? {
        val rmf = getRenderMP4FrameObject()
        return rmf?.let {
            InterpretationViewEffect.RenderMP4Frame(rmf)
        } ?: kotlin.run {
            null
        }
    }

    fun getRenderMP4FrameObject(): RenderMP4FrameObject? {
        return if (currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
            return RenderMP4FrameObject(numFrame = viewState.sopInstanceUIDBitmaps.size, bitmap = getCurrentFrameBitmap().adjustGammaCorrection(gammaValue), idFrame = currentFrameIndex, inforFrame = getInforFrame())
        } else null
    }


    fun getCurrentFileName(): String {
        return viewState.getShortSopInstanceUID()
    }

    fun getInforFrame(): String {
        return "${getCurrentFileName()}: ${currentFrameIndex + 1}/ ${numFrame}/ 0"
    }

    fun getCurrentFrameBitmap(): Bitmap {
        return viewState.sopInstanceUIDBitmaps.get(currentFrameIndex)
    }

    fun getGammaValue(): Double {
        return gammaValue
    }


}