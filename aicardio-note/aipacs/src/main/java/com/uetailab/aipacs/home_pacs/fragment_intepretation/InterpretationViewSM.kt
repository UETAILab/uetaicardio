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
import android.graphics.Canvas
import android.graphics.Matrix
import android.util.Log
import android.view.MotionEvent
import android.widget.CheckBox
import android.widget.ImageView
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewVM
import com.uetailab.aipacs.home_pacs.fragment_intepretation.DicomInterpretation.Companion.KEY_DIAGNOSIS
import org.json.JSONArray
import org.json.JSONObject

data class InterpretationViewState(
    val status: InterpretationViewStatus,
    val deviceID: String = "0968663886",
    val message: String,
    val studies: List<Int>,
    val studyID: String? = null,
    val studyInstanceUID: String? = null,
    val studyMetaData: JSONObject,
    val studyPreview: List<String>,
    val relativePath: String?=null,
    val bitmaps : List<Bitmap>,
    val studyInterpretation: JSONObject,
    val dicomInterpretation: DicomInterpretation,
    val machineInterpretation: DicomInterpretation,
    val dicomMetaData: JSONObject,
    val boundaryHeart: JSONArray,
    val efObject: EFObject,
    val glsValue: Float = 0.0F
) {

    companion object {
        // https://www.dicomlibrary.com/dicom/dicom-tags/
        const val HEX_CODE_NUMBER_FRAME = "NumberOfFrames"// "(0028,0008)"
        const val HEX_CODE_FRAME_DELAY = ""// "(0018,1066)"
        const val HEX_CODE_FRAME_TIME = "FrameTime" // "(0018,1063)"
        const val HEX_CODE_HEART_REATE = "HeartRate"// "(0018,1063)"
        const val HEX_CODE_PHYSICAL_DELTA_X = "PhysicalDeltaX" // "(0018,602C)" // (0008, 0090)
        const val HEX_CODE_PHYSICAL_DELTA_Y = "PhysicalDeltaY" // (0008, 0090)
        const val HEX_CODE_ROWS = "Rows"// "(0028,0010)"
        const val HEX_CODE_COLUMNS = "Columns"// "(0028,0011)"
        const val HEX_CODE_SIUID = "(0020,000D)"
        const val HEX_CODE_SopIUID = "(0008,0018)"
        const val LIST_FILE_DICOM = "ListFileDicom"

        fun getDicomMetaDataFromJSONObject(studyMetaData: JSONObject, relativePath: String): JSONObject {
            val listFileDicom = if (studyMetaData.has(LIST_FILE_DICOM)) studyMetaData.getJSONObject(LIST_FILE_DICOM) else JSONObject()
            val fileName = relativePath.substringAfterLast("____")
            listFileDicom.keys().forEach {
                if (it.contains(fileName)) {
                    try {
                        val obj = listFileDicom.getJSONObject(it)
                        return obj
                    } catch (e: Exception) {
                        Log.w(HomeViewVM.TAG, "getDicomMetaData: ${e}")
                        return JSONObject()
                    }
                }
            }
            return JSONObject()
        }



    }
    val dicomDiagnosis: DicomDiagnosis get() = dicomInterpretation.getDicomDiagnosisTest()

    val nColumn get() = if (dicomMetaData.has(HEX_CODE_COLUMNS)) dicomMetaData.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
    val nRow get() = if (dicomMetaData.has(HEX_CODE_ROWS)) dicomMetaData.getString(HEX_CODE_ROWS).toFloat() else 422.0F
    val listFileDicom: JSONObject get() = if (studyMetaData.has(LIST_FILE_DICOM)) studyMetaData.getJSONObject(LIST_FILE_DICOM) else JSONObject()
    val deltaX get() = if (dicomMetaData.has(HEX_CODE_PHYSICAL_DELTA_X)) dicomMetaData.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
    val deltaY get() = if (dicomMetaData.has(HEX_CODE_PHYSICAL_DELTA_Y)) dicomMetaData.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

    fun getDicomMetaData(relativePath: String): JSONObject{
        val fileName = relativePath.substringAfterLast("____")
        listFileDicom.keys().forEach {
            if (it.contains(fileName)) {
                try {
                    val obj = listFileDicom.getJSONObject(it)
                    return obj
                } catch (e: Exception) {
                    Log.w(HomeViewVM.TAG, "getDicomMetaData: ${e}")
                    return JSONObject()
                }
            }
        }
        return JSONObject()
    }


    fun getListStudyGVItem(): List<StudyGVItem> {
        return studyPreview.mapIndexed { index, s ->
            StudyGVItem(id = index.toLong(), name = filename(s), img_path = s )
        }
    }

    fun getListFrameCanvasRVItem(): List<FrameCanvasItem> {
        return bitmaps.mapIndexed { index, bitmap ->
            FrameCanvasItem(index,  getResizedBitmap(bitmap, 50, 35) )
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

    fun filename(fullPath: String): String { // gets filename without extension
        val dot: Int = fullPath.lastIndexOf("/")
        val sep: Int = fullPath.lastIndexOf(".jpg")
        return fullPath.substring(dot + 1, sep)
    }
}

sealed class InterpretationViewEffect {
    data class ShowToast(val message: String) : InterpretationViewEffect()
    data class RenderFragmentView(val fragmentViewObject: FragmentViewObject): InterpretationViewEffect()
    data class RenderToolUsing(val viewEvent: InterpretationViewEvent) : InterpretationViewEffect()
    data class RenderGridView(val relativePath: String?) : InterpretationViewEffect()
}
sealed class InterpretationViewEvent {
    object OnAutoCalculateEFGLS : InterpretationViewEvent()

    data class OnClickUndoClearDataAnnotation(val frameID: Int?=null, val keyClear: String?=null, val isClear: Boolean=false) : InterpretationViewEvent() // modClear = flase = undo

    data class RenderFragmentView(val buttonID: Int, val isLongClicked: Boolean, val isPlayingNextFrame: Boolean=true) : InterpretationViewEvent()
    data class FetchFileMP4(val studyID: String, val studyInstanceUID: String, val relativePath: String) : InterpretationViewEvent()
    data class FrameCanvasLongClicked(val frameCanvasItem: FrameCanvasItem) : InterpretationViewEvent()
    data class FrameCanvasClicked(val frameCanvasItem: FrameCanvasItem) : InterpretationViewEvent()
    data class OnToolClicked(val interpretationViewTool: InterpretationViewTool): InterpretationViewEvent()
    data class OnCheckBoxClicked(val interpretationViewTool: InterpretationViewTool): InterpretationViewEvent()


    data class OnTouchEvent(val view: InterpretationViewStudyPreviewCanvasView, val event: MotionEvent?, val ix: Float, val iy: Float) : InterpretationViewEvent()
    data class DrawModifyPoint(val touchEventObject: TouchEventObject) : InterpretationViewEvent()
    data class DrawModifyBoundary(val touchEventObject: TouchEventObject) : InterpretationViewEvent()
    data class RenderTouchDraw(val view: InterpretationViewStudyPreviewCanvasView, val canvas: Canvas?, val enableManualDraw: Boolean, val enableAutoDraw: Boolean) : InterpretationViewEvent()
    data class OnSaveDataDiagnosis(val dicomDiagnosis: DicomDiagnosis) : InterpretationViewEvent()
    object OnMeasureEFManual : InterpretationViewEvent()
    object OnSaveDataAnnotationToServer : InterpretationViewEvent()
}

sealed class InterpretationViewStatus {
    object Start : InterpretationViewStatus()
    object PassedDataFromHomeView : InterpretationViewStatus()
    object PassedDataFromInterpretationView : InterpretationViewStatus()
    object OnComponentClick : InterpretationViewStatus()
    object OnTouchDrawCanvas : InterpretationViewStatus()
    object OnPlayBackClick : InterpretationViewStatus()
    object OnUndoClearCanvas : InterpretationViewStatus()
    object OnSaveDataDiagnosis : InterpretationViewStatus()

    data class FetchedData(val viewEvent: InterpretationViewEvent) : InterpretationViewStatus()
    data class FetchedErrorData(val viewEvent: InterpretationViewEvent) : InterpretationViewStatus()
    data class OnFetchingData(val viewEvent: InterpretationViewEvent) : InterpretationViewStatus()
}

sealed class InterpretationViewTool(open val imageView: ImageView? = null, open val isLongClicked: Boolean = false){
    // isLongClicked: false: onClick,
    data class OnClickNoAction(override val imageView: ImageView?, override val isLongClicked: Boolean=false): InterpretationViewTool(imageView, isLongClicked) // default Tool
    data class OnClickZooming(override val imageView: ImageView?,override val isLongClicked: Boolean=false): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickDiagnosis(override val imageView: ImageView?,override val isLongClicked: Boolean=false, val dicomDiagnosis: DicomDiagnosis?=null): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickDrawPoint(override val imageView: ImageView?, override val isLongClicked: Boolean=false, val keyDrawPoint: String="point"): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickDrawBoundary(override val imageView: ImageView?, override val isLongClicked: Boolean=false, val keyDrawBoundary: String="boundary"): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickMeasureLength(override val imageView: ImageView?, override val isLongClicked: Boolean=false, val keyMearureLength: String="length"): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickMeasureArea(override val imageView: ImageView?, override val isLongClicked: Boolean=false, val keyMeasureArea: String="area"): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickCalculateEF(override val imageView: ImageView?,override val isLongClicked: Boolean=false): InterpretationViewTool(imageView, isLongClicked)
    data class OnClickShowBullEye(override val imageView: ImageView?,override val isLongClicked: Boolean=false): InterpretationViewTool(imageView, isLongClicked)

    data class OnClickCheckBox(val checkBox: CheckBox): InterpretationViewTool(null, false)
    data class  OnClickMeasureEF(override val imageView: ImageView?,override val isLongClicked: Boolean=false) : InterpretationViewTool(imageView, isLongClicked)
}