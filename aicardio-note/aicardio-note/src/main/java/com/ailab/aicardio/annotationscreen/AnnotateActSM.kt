package com.ailab.aicardio.annotationscreen

import android.graphics.Bitmap
import android.graphics.Canvas
import android.view.MotionEvent
import com.ailab.aicardio.*
import com.ailab.aicardio.annotationscreen.views.DrawCanvasView
import com.ailab.aicardio.repository.*
import com.imebra.DataSet
import org.json.JSONArray
import org.json.JSONObject

data class AnnotationViewState(
    val status: AnnotationViewStatus,
    val dataset: DataSet,
    val folderList: List<FolderItem>,
    val folder: String,
    val phone: String,
    val bitmaps: List<Bitmap>,
    val file: String,
    val dicomAnnotation: DicomAnnotation,
    val dicomDiagnosis: DicomDiagnosis,
    val tagsDicom : JSONObject,
    val machineAnnotation : DicomAnnotation,
    val machineDiagnosis: DicomDiagnosis,
    val boundaryHeart: JSONArray
) {

    fun getCurrentPosition(): Int {
        folderList.forEachIndexed { index, folderItem ->
            if (folderItem.path == file) return index
        }
        return -1
    }

    val sIUID get() =  if (tagsDicom.has(HEX_CODE_SIUID)) tagsDicom.getString(HEX_CODE_SIUID) else "empty_Study_Instance_UID"

    val sopIUID get() = if (tagsDicom.has(HEX_CODE_SopIUID))tagsDicom.getString(HEX_CODE_SopIUID) else "empty_SOP_Instance_UID"

    val nColumn get() = if (tagsDicom.has(HEX_CODE_COLUMNS)) tagsDicom.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
    val nRow get() = if (tagsDicom.has(HEX_CODE_ROWS)) tagsDicom.getString(HEX_CODE_ROWS).toFloat() else 422.0F
    val deltaX get() = if (tagsDicom.has(HEX_CODE_PHYSICAL_DELTA_X)) tagsDicom.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
    val deltaY get() = if (tagsDicom.has(HEX_CODE_PHYSICAL_DELTA_Y)) tagsDicom.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

}

sealed class AnnotationViewEffect {
    data class CopyAutoOneFrame(val message: String="") : AnnotationViewEffect()

    data class RenderAnnotationFrame(val renderAnnotation: RenderAnnotation) : AnnotationViewEffect()

    data class RenderButtonPlayPause(val button: Int, val isPlaying: Boolean) : AnnotationViewEffect()
    data class RenderButtonTool(val button: Int, val typeClicked: String) : AnnotationViewEffect()


    data class ShowSnackbar(val message: String) : AnnotationViewEffect()
    data class ShowToast(val message: String) : AnnotationViewEffect()
    data class LoadingProgress(val progress: Long) : AnnotationViewEffect()
    data class ShowDiagnosisDialog(val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEffect()
    data class RenderDiagnosisTool(val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEffect()

    data class ShowSaveDialog(val file: String, val bitmaps: List<Bitmap>, val dicomAnnotation: DicomAnnotation, val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEffect()
    class ShowUserLoginDialog : AnnotationViewEffect()
    data class Test(val message: String) : AnnotationViewEffect()
    data class TestAsync(val message: String) : AnnotationViewEffect()
    data class CopyAutoAllFrames(val message: String="") : AnnotationViewEffect()
    data class RenderLoginButton(val phone: String="LOGIN"): AnnotationViewEffect()
}

sealed class AnnotationViewEvent {


    data class FetchNewsFile(val file: String) : AnnotationViewEvent()
    data class FetchNewsFolder(val folder: String) : AnnotationViewEvent()

    data class OnToolUsing(val toolId: Int, val typeClicked: String, val isPlaying: Boolean = false) : AnnotationViewEvent()
    data class ProcessTouchEvent(
        val view: DrawCanvasView,
        val event: MotionEvent?,
        val ix: Float,
        val iy: Float
    ) : AnnotationViewEvent()

    data class NewsItemFileClicked(val folderItem: FolderItem) : AnnotationViewEvent()
    data class NewsItemFileLongClicked(val folderItem: FolderItem) : AnnotationViewEvent()


    data class RenderDraw(val view: DrawCanvasView, val canvas: Canvas?, val enableManualDraw: Boolean = true, val enableAutoDraw: Boolean = false) : AnnotationViewEvent()
    data class SaveDiagnosis(val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEvent()

    data class FetchNewsBothFolderAndFile(val folder: String?, val file: String?) : AnnotationViewEvent()

    data class OnSaveUserLogin(val user: User) : AnnotationViewEvent()

    class NewsFrameClicked(val frameItem: FrameItem) : AnnotationViewEvent()
    class NewsFrameLongClicked(val frameItem: FrameItem) : AnnotationViewEvent()

    data class FolderFetchedError(val result: LCE.Result<List<FolderItem>>) : AnnotationViewEvent()

//    data class FolderFetchedError(val result: LCE.Error<List<FolderItem>>) : AnnotationViewEvent()

    data class FolderFetchedSuccess(val result: LCE.Result<List<FolderItem>>) : AnnotationViewEvent()
    data class FileFetchedError(val result: LCE.Result<DicomObject>) : AnnotationViewEvent()

    data class FileFetchedSuccess(
        val viewState: AnnotationViewState,
        val viewEffect: AnnotationViewEffect.ShowToast
    ) : AnnotationViewEvent()
    data class AnnotationAndDiagnosis(val result: LCE.Result<DicomObject>, val file: String) : AnnotationViewEvent()
    data class NewProgess(val progress: Long) : AnnotationViewEvent()

    data class DrawPoint(val obj: TouchEventObject): AnnotationViewEvent()
    data class ModifyDrawPoint(val obj: TouchEventObject): AnnotationViewEvent()
    data class DrawBoundary(val obj: TouchEventObject): AnnotationViewEvent()
    data class ModifyDrawBoundary(val obj: TouchEventObject): AnnotationViewEvent()

    data class UndoPoint(val key: String, val frameIdx: Int) : AnnotationViewEvent()
    data class UndoBoundary(val key: String, val frameIdx: Int) : AnnotationViewEvent()


    data class ClearPoint(val key: String, val frameIdx: Int) : AnnotationViewEvent()
    data class ClearBoundary(val key: String, val frameIdx: Int) : AnnotationViewEvent()

    object NextFrame : AnnotationViewEvent()
    object PlayPauseVideo : AnnotationViewEvent()
    object ShowNextFrame : AnnotationViewEvent()
    object ShowPreviousFrame : AnnotationViewEvent()
    object ShowLastFrame : AnnotationViewEvent()
    object ShowFirstFrame : AnnotationViewEvent()

    object UndoAnnotation : AnnotationViewEvent()
    object ClearAnnotation : AnnotationViewEvent()
    data class ShowEsvEdvOrAnnotationFrame(val isEsvEdv: Boolean = false) : AnnotationViewEvent()

    object OnSaveData : AnnotationViewEvent()
    object OnSaveConfirmed : AnnotationViewEvent()
    object OnSaveDataToServer : AnnotationViewEvent()
    data class OnSaveDataToDisk(val pushNotification: Boolean = false) : AnnotationViewEvent()

    data class SaveDataToDiskError(
        val viewState: AnnotationViewState,
        val viewEffect: AnnotationViewEffect.ShowToast
    ) : AnnotationViewEvent()

    data class SaveDataToDiskSuccess(
        val viewState: AnnotationViewState,
        val viewEffect: AnnotationViewEffect.ShowToast?
    ) : AnnotationViewEvent()

    data class SaveDataToServerError(
        val viewState: AnnotationViewState,
        val viewEffect: AnnotationViewEffect.ShowToast
    ) : AnnotationViewEvent()

    data class SaveDataToServerSuccess(
        val viewState: AnnotationViewState,
        val viewEffect: AnnotationViewEffect.ShowToast
    ) : AnnotationViewEvent()

    data class Test(val message: String) : AnnotationViewEvent()
    data class TestAsync(val message: String) : AnnotationViewEvent()

//    data class SaveDataToDiskError(val readResult: LCE.Result<String>) : AnnotationViewEvent()

    object BackToParentFolder : AnnotationViewEvent()
    object OnUserLogin : AnnotationViewEvent()
    object ClearCurrentTool : AnnotationViewEvent()
    object AutoAnalysis : AnnotationViewEvent()
    object AutoAnalysisAsyncError : AnnotationViewEvent()
    object CopyAutoOneFrame : AnnotationViewEvent()
    object CopyAutoAllFrames : AnnotationViewEvent()



    data class AutoAnalysisAsyncSuccess(
        val dicomAnnotation: DicomAnnotation,
        val dicomDiagnosis: DicomDiagnosis
    ) : AnnotationViewEvent()

    data class ToggleAutoDraw(val isAutoDraw: Boolean) : AnnotationViewEvent()
    data class ToggleManualDraw(val isManualDraw: Boolean) : AnnotationViewEvent()

}

sealed class AnnotationViewStatus {
    object FolderFetching : AnnotationViewStatus()
    object FolderFetched : AnnotationViewStatus()
    object FolderNotFetched : AnnotationViewStatus()

    object Fetching : AnnotationViewStatus()
    object Fetched : AnnotationViewStatus()
    object NotFetched : AnnotationViewStatus()


    object SavingToDisk : AnnotationViewStatus()
    object SavedToDiskSuccess : AnnotationViewStatus()
    object SavedToDiskError : AnnotationViewStatus()

    object SavedToServerSuccess : AnnotationViewStatus()
    object SavedToServerError : AnnotationViewStatus()

    data class DrawPoint(val frameIdx: Int) : AnnotationViewStatus()
    data class DrawBoundary(val frameIdx: Int) : AnnotationViewStatus()
    object ModifyDrawPoint : AnnotationViewStatus()
    object ModifyDrawBoundary : AnnotationViewStatus()

    data class OpenAnnotationActivity(val folder: String) : AnnotationViewStatus()
    data class Test(val message: String) : AnnotationViewStatus()
    data class TestAsync(val message: String) : AnnotationViewStatus()

    object SetESVState : AnnotationViewStatus()
    object SetEDVState : AnnotationViewStatus()

    object DiagnosisEntered : AnnotationViewStatus()
    object LoginDone : AnnotationViewStatus()
    object UndoPoint : AnnotationViewStatus()
    object UndoBoundary : AnnotationViewStatus()

    object ClearPoint : AnnotationViewStatus()
    object ClearBoundary : AnnotationViewStatus()
    object AutoAnnalysisFetched : AnnotationViewStatus()
    object AutoAnnalysisFetching : AnnotationViewStatus()
    object CopyAutoOneFrame : AnnotationViewStatus()
    object CopyAutoAllFrames : AnnotationViewStatus()



}