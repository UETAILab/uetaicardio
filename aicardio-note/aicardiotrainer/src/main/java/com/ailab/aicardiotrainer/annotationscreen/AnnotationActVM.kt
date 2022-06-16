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

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.ailab.aicardiotrainer.annotationscreen.AnnotationActReducer
import com.ailab.aicardiotrainer.annotationscreen.AnnotationViewStatus
import com.ailab.aicardiotrainer.repositories.*
import com.imebra.DataSet

import com.rohitss.aacmvi.AacMviViewModel
import org.json.JSONObject
import java.io.File

class AnnotationActVM(applicationAnnotate: Application) : AacMviViewModel<AnnotationViewState, AnnotationViewEffect, AnnotationViewEvent>(applicationAnnotate) {

    companion object {

        const val TAG = "AnnotationActVM"
        const val DICOM_TAG = "dicom_tags"
        const val PHONE = "deviceID"
        const val FILE = "path_file"
        const val MANUAL_DIAGNOSIS = "dicomDiagnosis"
        const val MANUAL_ANNOTATION = "dicomAnnotation"
        const val MACHINE_ANNOTATION = "machineAnnotation"
        const val MACHINE_DIAGNOSIS= "machineDiagnosis"
        const val KEY_VERSION = "VERSION"
        const val VERSION_NUMBER = "4.0"
    }


    init {
        viewState = AnnotationViewState(
            status = AnnotationViewStatus.Start,
            dataset = DataSet(),
//            folderList = emptyList(),
            folder = "",
            phone = User.getPhone(context = applicationAnnotate) ?: User.DEFAULT_PHONE,
            bitmaps = emptyList(),
            file = "",
            dicomAnnotation = DicomAnnotation(),
            dicomDiagnosis = DicomDiagnosis(),
            tagsDicom = JSONObject(),
            machineAnnotation = DicomAnnotation(),
            machineDiagnosis = DicomDiagnosis()
        )
    }
    val numFrame get() =  viewState.bitmaps.size

    private var currentFrameIndex: Int = 0
    private var isPlaying: Boolean = true

    private var currentToolId: Int? = null
    private var isGls: Boolean = false // check value at Perimeter checkbox, if it true then draw for gls else ef

    private var enableManualDraw: Boolean = true
    private var enableAutoDraw: Boolean = false

    private var toolClickedType: String? = null
    private var modifyPointIndex = -1

    private var modifyBoundaryIndex = Pair(-1, -1) // (index of boundary, index of point)
    private val isValidFrameState get() = (isPlaying == false && numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame)

    private val isESV get() = viewState.dicomAnnotation.getIsESVWithFrameIndex(currentFrameIndex)
    private val isEDV get() = viewState.dicomAnnotation.getIsEDVWithFrameIndex(currentFrameIndex)

    private val lengthCalculator get() = viewState.dicomAnnotation.getMeasureByKey(currentFrameIndex, DicomAnnotation.LENGTH)
    private val areaCalculator get() = viewState.dicomAnnotation.getMeasureByKey(currentFrameIndex, DicomAnnotation.AREA)
    private val volumeCalculator get() = viewState.dicomAnnotation.getMeasureByKey(currentFrameIndex, DicomAnnotation.VOLUME)
    val nPointEF get() = viewState.dicomAnnotation.getPointArray(currentFrameIndex, DicomAnnotation.EF_POINT).length()
    val nPointGLS get() = viewState.dicomAnnotation.getPointArray(currentFrameIndex, DicomAnnotation.GLS_POINT).length()
    val textEsvEDV get() = viewState.dicomAnnotation.getEsvEDVTextDraw()


    private val keyModEfGls get() = if (isGls == true) DicomAnnotation.MOD_GLS else DicomAnnotation.MOD_EF

    val keyPoint get() = "${keyModEfGls}${DicomAnnotation.MOD_POINT}"
    val keyBoundary get() = "${keyModEfGls}${DicomAnnotation.MOD_BOUNDARY}"

    fun getIsValidFrameState(): Boolean {
        return isValidFrameState
    }

    fun getCurrentTool(): Pair<Int?, String?> {
        return Pair(currentToolId, toolClickedType)
    }

    fun reduce(reducer: AnnotationActReducer) {
        val result = reducer.reduce()
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }

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


    override fun process(viewEvent: AnnotationViewEvent) {
        super.process(viewEvent)
        when (viewEvent) {

//            is com.ailab.aicardio.annotationscreen.AnnotationViewEvent.NewsItemFileClicked -> newsItemFileClicked(viewEvent.folderItem)
//            is com.ailab.aicardio.annotationscreen.AnnotationViewEvent.NewsItemFileLongClicked -> newsItemFileClicked(viewEvent.folderItem)

            is AnnotationViewEvent.NewsFrameClicked -> newsFrameClicked(viewEvent.frameItem)

        }
    }

    private fun newsFrameClicked(frameItem: FrameItem) {
        currentFrameIndex = frameItem.index
        getRenderAnnotationFrame()?.let { viewEffect = it }
    }


    fun getRenderAnnotationFrame(): AnnotationViewEffect.RenderAnnotationFrame? {
//        if (numFrame > 0) return AnnotationViewEffect.RenderAnnotationFrame(RenderAnnotation())
//        Log.w("getRenderAnnotationFrame", "c: $currentFrameIndex #: $numFrame")

        return if (currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
            val o = viewState.dicomAnnotation.getFrameAnnotationObj(currentFrameIndex)
            val ro = RenderAnnotation(nPointGLS = nPointGLS, nPointsEF = nPointEF, ef = EFObject.convertToEFObject(o), esvEdvText = textEsvEDV,
                length = lengthCalculator, area =  areaCalculator, volume = volumeCalculator,
                isPlaying = isPlaying, numFrame = numFrame, indexFrame = currentFrameIndex,
                infoText = getInfoText(), bitmap = getCurrentFrameBitmap(), isESV=isESV, isEDV=isEDV)
            return AnnotationViewEffect.RenderAnnotationFrame(ro)
        } else null
    }

    private fun getCurrentFrameBitmap(): Bitmap {
        return viewState.bitmaps.get(currentFrameIndex)
    }

    fun getInfoText(): String {
        return "${viewState.dicomAnnotation.getNFrameAnnotated()} / ${currentFrameIndex} / ${numFrame - 1} / ${File(viewState.file).name}"

    }

    fun getFileName(): String {
        return viewState.file
    }

    fun getDiagnosis(): DicomDiagnosis {
        return viewState.dicomDiagnosis
    }

    fun getIsPlaying(): Boolean {
        return isPlaying
    }

    fun getListFrameList(): List<FrameItem> {
        return viewState.bitmaps.mapIndexed { index, bitmap ->
            FrameItem(index,  getResizedBitmap(bitmap, 50, 35) )
        }
    }

    fun getResizedBitmap(bm: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        val width = bm.width
        val height = bm.height
        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height
        // CREATE A MATRIX FOR THE MANIPULATION
        val matrix = Matrix()
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight)

        // "RECREATE" THE NEW BITMAP
        return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false)
    }

    fun hasNoLabel(): Boolean {
        return (viewState.dicomDiagnosis.chamberIdx == -1)
    }

    fun setIsGls(value: Boolean) {
        this.isGls = value
    }


    fun getIsGls(): Boolean? {
        return isGls
    }

    fun setIsPlaying(isPlayingValue: Boolean) {
        isPlaying = isPlayingValue
    }

    fun setCurrentToolId(toolId: Int?) {
        this.currentToolId = toolId
    }

    fun setToolClickedType(typeClicked: String) {
        this.toolClickedType = typeClicked
    }

    fun getCurrentFrameIndex(): Int {
        return currentFrameIndex
    }

    fun hasEFBoundaryAndPoint(frameIdx: Int): Boolean {
//        Log.w(TAG, "hasEFBoundaryAndPoint -- ${frameIdx}")
        return viewState.dicomAnnotation.hasBoundaryAndPoint(frameIdx, key_point = DicomAnnotation.EF_POINT, key_boundary = DicomAnnotation.EF_BOUNDARY)
    }

    fun hasGLSBoundaryAndPoint(frameIdx: Int): Boolean {
        return viewState.dicomAnnotation.hasBoundaryAndPoint(frameIdx, key_point = DicomAnnotation.GLS_POINT, key_boundary = DicomAnnotation.GLS_BOUNDARY)
    }

    fun setCurrentFrameIndex(frameIdx: Int) {
//        Log.w("setCurrentFrameIndex", "$frameIdx")
        if (numFrame > 0)
            currentFrameIndex = frameIdx
        else currentFrameIndex = -1
    }

    fun forceCurrentFrameIndex(frameIdx: Int) {
        currentFrameIndex = frameIdx
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


    fun reduce(result: AnnotationStateEffectObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }

}
