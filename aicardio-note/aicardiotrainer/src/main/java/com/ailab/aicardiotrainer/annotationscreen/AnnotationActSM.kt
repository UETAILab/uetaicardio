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

import android.graphics.Bitmap
import android.graphics.Canvas
import android.view.MotionEvent
import com.ailab.aicardiotrainer.repositories.FrameItem
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.repositories.*
import com.imebra.DataSet
import org.json.JSONObject

data class AnnotationViewState(
    val status: AnnotationViewStatus,
    val dataset: DataSet,
    val folder: String,
    val phone: String,
    val bitmaps: List<Bitmap>,
    val file: String,
    val tagsDicom : JSONObject,

    val dicomAnnotation: DicomAnnotation,
    val dicomDiagnosis: DicomDiagnosis,

    val machineAnnotation : DicomAnnotation,
    val machineDiagnosis: DicomDiagnosis
)

sealed class AnnotationViewStatus {
    object Start: AnnotationViewStatus()
    object DownloadingDicom : AnnotationViewStatus()
    object DownloadedDicom : AnnotationViewStatus()
    object OnReadingDicom: AnnotationViewStatus()
    object ReadingDicomError : AnnotationViewStatus()
    object ReadingDicomSuccess : AnnotationViewStatus()
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

sealed class AnnotationViewEffect {
    data class ShowToast(val message: String) : AnnotationViewEffect()
    data class CopyAutoOneFrame(val message: String="") : AnnotationViewEffect()

    data class RenderAnnotationFrame(val renderAnnotation: RenderAnnotation) : AnnotationViewEffect()

    data class RenderButtonPlayPause(val button: Int, val isPlaying: Boolean) : AnnotationViewEffect()
    data class RenderButtonTool(val button: Int, val typeClicked: String) : AnnotationViewEffect()


    data class ShowSnackbar(val message: String) : AnnotationViewEffect()
    data class LoadingProgress(val progress: Long) : AnnotationViewEffect()
//    data class ShowDiagnosisDialog(val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEffect()
//    data class RenderDiagnosisTool(val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEffect()

//    data class ShowSaveDialog(val file: String, val bitmaps: List<Bitmap>, val dicomAnnotation: DicomAnnotation, val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEffect()
    class ShowUserLoginDialog : AnnotationViewEffect()
    data class Test(val message: String) : AnnotationViewEffect()
    data class TestAsync(val message: String) : AnnotationViewEffect()
    data class CopyAutoAllFrames(val message: String="") : AnnotationViewEffect()
    data class RenderLoginButton(val phone: String="LOGIN"): AnnotationViewEffect()

}

sealed class AnnotationViewEvent {
    data class DownloadDicom(val dicomJPGPath: String): AnnotationViewEvent()
    class ReadDicom(val dicomPath: String): AnnotationViewEvent()
    class ReadingDicomProgess(val progress: Long) : AnnotationViewEvent()
    class FileFetchedError(val resultLaunch: LCE.Result<DicomObject>) : AnnotationViewEvent()
    class FileFetchedSuccess(val annotationStateEffectObject: AnnotationStateEffectObject ) : AnnotationViewEvent()

    data class FetchNewsFile(val file: String) : AnnotationViewEvent()
    data class FetchNewsFolder(val folder: String) : AnnotationViewEvent()

//    data class FileFetchedSuccess(
//        val viewState: AnnotationViewState,
//        val viewEffect: AnnotationViewEffect.ShowToast
//    ) : AnnotationViewEvent()
//
    data class OnToolUsing(val toolId: Int, val typeClicked: String, val isPlaying: Boolean = false) : AnnotationViewEvent()
    data class ProcessTouchEvent(
        val view: DrawCanvasView,
        val event: MotionEvent?,
        val ix: Float,
        val iy: Float
    ) : AnnotationViewEvent()

//    data class NewsItemFileClicked(val folderItem: FolderItem) : AnnotationViewEvent()
//    data class NewsItemFileLongClicked(val folderItem: FolderItem) : AnnotationViewEvent()


    data class RenderDraw(val view: DrawCanvasView, val canvas: Canvas?, val enableManualDraw: Boolean = true, val enableAutoDraw: Boolean = false) : AnnotationViewEvent()
    data class SaveDiagnosis(val dicomDiagnosis: DicomDiagnosis) : AnnotationViewEvent()

    data class FetchNewsBothFolderAndFile(val folder: String?, val file: String?) : AnnotationViewEvent()

    data class OnSaveUserLogin(val user: User) : AnnotationViewEvent()

    class NewsFrameClicked(val frameItem: FrameItem) : AnnotationViewEvent()
    class NewsFrameLongClicked(val frameItem: FrameItem) : AnnotationViewEvent()

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
