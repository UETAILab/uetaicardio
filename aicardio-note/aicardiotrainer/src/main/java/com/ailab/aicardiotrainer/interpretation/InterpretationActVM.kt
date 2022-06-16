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

import aacmvi.AacMviViewModel
import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import java.lang.Math.round
import kotlin.math.max

class InterpretationActVM(applicationAnnotate: Application) : AacMviViewModel<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent>(applicationAnnotate) {

    companion object {
        const val TOOL_NO_ACTION = "NO_ACTION"
        const val TAG = "InterpretationActVM"
        const val TOOL_DRAW_POINT = "DRAW_POINT"
        const val TOOL_DRAW_BOUNDARY = "DRAW_BOUNDARY"
        const val TOOL_MEASURE_LENGTH = "MEASURE_LENGTH"
        const val TOOL_MEASURE_AREA = "MEASURE_AREA"

        const val TOOL_CHECK_BOX_PERIMETER = "CHECK_BOX_PERIMETER"
        const val TOOL_CHECK_BOX_MANUAL = "CHECK_BOX_MANUAL"
        const val TOOL_CHECK_BOX_ESV = "CHECK_BOX_ESV"
        const val TOOL_CHECK_BOX_EDV = "CHECK_BOX_EDV"

        const val TYPE_DRAW_GLS = "DRAW_GLS"
        const val TYPE_DRAW_EF = "DRAW_EF"
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
            sopInstanceUIDTags = JSONObject(),
            dicomAnnotation = DicomAnnotation(),
            dicomDiagnosis = DicomDiagnosis(),
            tagsDicom = JSONObject(),
            machineAnnotation = DicomAnnotation(),
            machineDiagnosis = DicomDiagnosis(),
            gls = emptyList<Float>()
        )
    }

    val numFrame get() =  viewState.sopInstanceUIDBitmaps.size

    private var currentFrameIndex: Int = 0
    private var isPlaying: Boolean = true
    private var gammaValue = 1.0

    private var toolNameUsing : String = ""
    private var toolClickedType : Boolean = false // false: short clicked/, true: long clicked
    private var modifyPointIndex = -1

    private var modifyBoundaryIndex = Pair(-1, -1) // (index of boundary, index of point)

    private var isGls: Boolean = false // check value at Perimeter checkbox, if it true then draw for gls else ef
    private val isESV get() = viewState.dicomAnnotation.getIsESVWithFrameIndex(currentFrameIndex)
    private val isEDV get() = viewState.dicomAnnotation.getIsEDVWithFrameIndex(currentFrameIndex)

    private val keyModEfGls get() = if (isGls == true) DicomAnnotation.MOD_GLS else DicomAnnotation.MOD_EF

    val keyPoint get() = "${keyModEfGls}${DicomAnnotation.MOD_POINT}"
    val keyBoundary get() = "${keyModEfGls}${DicomAnnotation.MOD_BOUNDARY}"

    private var enableManualDraw: Boolean = true
    private var enableAutoDraw: Boolean = false

    fun getModifyPointIndex(): Int {
        return modifyPointIndex
    }
    fun setModifyPointIndex(value: Int) {
        modifyPointIndex = value
    }

    fun setModifyBoundaryIndex(value: Pair<Int, Int>) {
        modifyBoundaryIndex = value
    }

    fun getModifyBoundaryIndex(): Pair<Int, Int> {
        return modifyBoundaryIndex
    }

    fun setAutoDraw(autoDrawMachine: Boolean) {
        enableAutoDraw = autoDrawMachine
    }

    fun setManualDraw(manualDraw: Boolean) {
        enableManualDraw = manualDraw
    }

    fun getEnableManualDraw(): Boolean {
        return enableManualDraw
    }

    fun getEnableAutoDraw(): Boolean {
        return enableAutoDraw
    }



    fun setToolNameUsing(toolName: String, typeClicked: Boolean = false) {
        this.toolNameUsing = toolName
        this.toolClickedType = typeClicked
    }

    fun setToolClickedType(typeClicked: Boolean) {
        this.toolClickedType = typeClicked
    }


    private val isValidFrameState get() = (isPlaying == false && numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame)

    fun getIsValidFrameState(): Boolean {
        return isValidFrameState
    }

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
            return RenderMP4FrameObject(numFrame = viewState.sopInstanceUIDBitmaps.size, bitmap = getCurrentFrameBitmap().adjustGammaCorrection(gammaValue),
                idFrame = currentFrameIndex, inforFrame = getInforFrame(), isESV=isESV, isEDV=isEDV)
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

    fun getToolNameUsing(): String {
        return toolNameUsing
    }

    fun getToolClickedType(): Boolean {
        return toolClickedType
    }

    fun hasNoLabel(): Boolean {
        return false
        return (viewState.dicomDiagnosis.chamberIdx == -1)
    }

    fun getIsGls(): Boolean? {
        return isGls
    }

    fun setIsGls(toolTypeClick: Boolean) {
        isGls = toolTypeClick
    }


}