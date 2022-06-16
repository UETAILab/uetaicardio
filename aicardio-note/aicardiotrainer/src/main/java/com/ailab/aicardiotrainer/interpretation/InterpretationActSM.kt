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

import android.graphics.Bitmap
import android.graphics.Canvas
import android.view.MotionEvent
import com.ailab.aicardiotrainer.studyscreen.DicomItem
import org.json.JSONObject

data class InterpretationViewState(
    val status: InterpretationViewStatus,

    // data for study
    val studyInstanceUIDPath: String,
    val studyFiles: List<String>,
    val studyRepresentation: List<Bitmap>,
    val studyHashMap: HashMap<String, Bitmap>,

    // data for file dicom
    val sopInstanceUIDPath: String,
    val sopInstanceUIDBitmaps: List<Bitmap>,
    val sopInstanceUIDTags: JSONObject,
    val dicomAnnotation: DicomAnnotation,
    val machineAnnotation: DicomAnnotation,
    val dicomDiagnosis: DicomDiagnosis,
    val machineDiagnosis: DicomDiagnosis,
    val tagsDicom: JSONObject,
    val gls: List<Float>

) {
    companion object {
        const val newWidthBitmap = 50
        const val newHeightBitmap = 35
        // https://www.dicomlibrary.com/dicom/dicom-tags/
        const val HEX_CODE_NUMBER_FRAME = "(0028,0008)"
        const val HEX_CODE_FRAME_DELAY = "(0018,1066)"
        const val HEX_CODE_FRAME_TIME = "(0018,1063)"
        const val HEX_CODE_PHYSICAL_DELTA_X = "(0018,602C)" // (0008, 0090)
        const val HEX_CODE_PHYSICAL_DELTA_Y = "(0018,602E)" // (0008, 0090)
        const val HEX_CODE_ROWS = "(0028,0010)"
        const val HEX_CODE_COLUMNS = "(0028,0011)"
        const val HEX_CODE_SIUID = "(0020,000D)"
        const val HEX_CODE_SopIUID = "(0008,0018)"


    }

    val sIUID get() =  if (tagsDicom.has(HEX_CODE_SIUID)) tagsDicom.getString(HEX_CODE_SIUID) else "empty_Study_Instance_UID"

    val sopIUID get() = if (tagsDicom.has(HEX_CODE_SopIUID))tagsDicom.getString(HEX_CODE_SopIUID) else "empty_SOP_Instance_UID"

    val nColumn get() = if (tagsDicom.has(HEX_CODE_COLUMNS)) tagsDicom.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
    val nRow get() = if (tagsDicom.has(HEX_CODE_ROWS)) tagsDicom.getString(HEX_CODE_ROWS).toFloat() else 422.0F
    val deltaX get() = if (tagsDicom.has(HEX_CODE_PHYSICAL_DELTA_X)) tagsDicom.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
    val deltaY get() = if (tagsDicom.has(HEX_CODE_PHYSICAL_DELTA_Y)) tagsDicom.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

    fun getListSopInstanceUIDItem(): List<SopInstanceUIDItem> {
        return  this.studyFiles.mapIndexed { index, s ->
//            SopInstanceUIDItem(id = index.toLong(), name = filename(s), imgPath = s, bitmap = getResizedBitmap(studyRepresentation.get(index), newWidthBitmap, newHeightBitmap) )
            SopInstanceUIDItem(id = index.toLong(), name = filename(s), imgPath = s, bitmap = studyRepresentation.get(index) )
        }
    }

    fun filename(fullPath: String): String { // gets filename without extension
        // "/storage/emulated/0/Download/StudyInstanceUID/SOPInstanceUID____fileName.jpg"
        val dot: Int = fullPath.lastIndexOf("/")
        val sep: Int = fullPath.lastIndexOf(".jpg")
        return fullPath.substring(dot + 1, sep)
    }

    fun getShortSopInstanceUID(): String {
        val shortNameArr = sopInstanceUIDPath.split("____")
        val result = shortNameArr[shortNameArr.size - 1]

        if (result.contains(".")) {
            val sep: Int = result.lastIndexOf(".")
            return result.substring(0, sep)
        }

        return shortNameArr[shortNameArr.size - 1]
    }

    fun getSubmitListFrameItem(): List<FrameItem> {
        return this.sopInstanceUIDBitmaps.mapIndexed { index, bitmap ->
            FrameItem(index=index, bitmap = bitmap)
        }
    }


}

sealed class InterpretationViewStatus {
    object OnLoadingMP4File: InterpretationViewStatus()
    object OnLoadedMP4File: InterpretationViewStatus()
    object UndoPoint : InterpretationViewStatus()
    object UndoBoundary : InterpretationViewStatus()

    object ClearPoint : InterpretationViewStatus()
    object ClearBoundary : InterpretationViewStatus()
    object Start: InterpretationViewStatus()

//    data class OnDownloadingStudyInstanceUID(val studyInstanceUID: String): InterpretationViewStatus()

//    object OnDoneDownloadingStudyInstanceUID: InterpretationViewStatus()
//    object OnDownloadingStudyInstanceUID: InterpretationViewStatus()

    object OnLoadingRepresentationStudyInstanceUID: InterpretationViewStatus()
    object OnDoneLoadingRepresentationStudyInstanceUID: InterpretationViewStatus()

    object OnLoadingMP4SopInstanceUID: InterpretationViewStatus()
    object OnDoneLoadingMP4SopInstanceUID: InterpretationViewStatus()
    object EnhanceContrastBitmap : InterpretationViewStatus()

    data class DrawPoint(val frameIdx: Int) : InterpretationViewStatus()
    data class DrawBoundary(val frameIdx: Int) : InterpretationViewStatus()
    object ModifyDrawPoint : InterpretationViewStatus()
    object ModifyDrawBoundary : InterpretationViewStatus()
    object SetESVState : InterpretationViewStatus()
    object SetEDVState : InterpretationViewStatus()
    object OnGLSCalculating : InterpretationViewStatus()
    object DoneGLSCalculating : InterpretationViewStatus()

}

sealed class InterpretationViewEffect {
    data class ShowToast(val message: String) : InterpretationViewEffect()
    data class RenderMP4Frame(val rmf: RenderMP4FrameObject) : InterpretationViewEffect()
    data class RenderComponentActivity(val idComponent: Int, val isPlaying: Boolean) : InterpretationViewEffect()
    data class RenderTextViewGammaCorrection(val gamma: Double): InterpretationViewEffect()
    class RenderButtonTool(val viewEvent: InterpretationViewEvent.OnToolUsing) : InterpretationViewEffect()
    class RenderClearCurrentTool(val viewEvent: InterpretationViewEvent.ClearCurrentTool) : InterpretationViewEffect()

}
sealed class InterpretationViewEvent {

    data class LoadingMP4File(val filePath: String) : InterpretationViewEvent()
    object ClearCurrentTool : InterpretationViewEvent()

    object UndoAnnotation : InterpretationViewEvent()
    object ClearAnnotation : InterpretationViewEvent()

    data class UndoPoint(val key: String, val frameIdx: Int) : InterpretationViewEvent()
    data class UndoBoundary(val key: String, val frameIdx: Int) : InterpretationViewEvent()


    data class ClearPoint(val key: String, val frameIdx: Int) : InterpretationViewEvent()
    data class ClearBoundary(val key: String, val frameIdx: Int) : InterpretationViewEvent()

    object NextFrame : InterpretationViewEvent()
    object ShowNextFrame : InterpretationViewEvent()

    object ShowPreviousFrame : InterpretationViewEvent()
    object ShowFirstFrame : InterpretationViewEvent()
    object ShowLastFrame : InterpretationViewEvent()
    object PlayPauseVideo : InterpretationViewEvent()

    data class PlayBackMP4File(val fileMP4Path: String): InterpretationViewEvent()

    data class LoadingRepresentationStudyInstanceUID(val studyInstanceUID: String) : InterpretationViewEvent()
    data class LoadingMP4SopInstanceUID(val item: SopInstanceUIDItem) : InterpretationViewEvent()
    data class SopInstanceUIDFrameClicked(val frameItem: FrameItem) : InterpretationViewEvent()
    data class SopInstanceUIDFrameLongClicked(val frameItem: FrameItem) : InterpretationViewEvent()
    data class EnhanceContrastBitmap(val threshold: Int) : InterpretationViewEvent()
    data class OnChangeGammaCorrection(val threshold: Int) : InterpretationViewEvent()


    data class DrawPoint(val obj: TouchEventObject): InterpretationViewEvent()
    data class ModifyDrawPoint(val obj: TouchEventObject): InterpretationViewEvent()
    data class DrawBoundary(val obj: TouchEventObject): InterpretationViewEvent()
    data class ModifyDrawBoundary(val obj: TouchEventObject): InterpretationViewEvent()


    data class ProcessTouchEvent(
        val view: InterpretationCanvasView,
        val event: MotionEvent?,
        val ix: Float,
        val iy: Float
    ) : InterpretationViewEvent()

    data class OnToolUsing(val toolName: String, val toolTypeClick: Boolean, val toolButtonID: Int, val isPlaying: Boolean = false) : InterpretationViewEvent()
    class RenderDraw(val view: InterpretationCanvasView, val canvas: Canvas?,  val enableManualDraw: Boolean = true, val enableAutoDraw: Boolean = false) : InterpretationViewEvent()
    object ShowBullEyeMapping : InterpretationViewEvent()

}

