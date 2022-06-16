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

package com.ailab.aicardiotrainer.studyscreen

import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.api.ProgressListener
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.internal.format

class StudyLoadDicomMVI(val listener: ProgressListener) {

    companion object {

        const val TAG = "StudyLoadDicomMVI"

//        // For Singleton instantiation
//        @Volatile
//        private var instance: StudyLoadDicomMVI? = null
//
//        fun getInstance() =
//            instance ?: synchronized(this) {
//                instance
//                    ?: StudyLoadDicomMVI()
//                        .also { instance = it }
//            }
//
//
//        fun process(viewModel: StudyActVM, viewEvent: StudyViewEvent) {
//            getInstance().process(viewModel, viewEvent)
//        }
//
//        fun renderViewState(studyActivity: StudyActivity, viewState: StudyViewState) {
//            getInstance().renderViewState(studyActivity, viewState)
//        }
//
//        fun renderViewEffect(studyActivity: StudyActivity, viewEffect: StudyViewEffect) {
//            getInstance().renderViewEffect(studyActivity, viewEffect)
//        }

    }

    private val studyScreenRepository = StudyScreenRepository.getInstance()

     fun renderViewEffect(studyActivity: StudyActivity, viewEffect: StudyViewEffect) {
    }

     fun renderViewState(studyActivity: StudyActivity, viewState: StudyViewState) {

        when (viewState.status) {
            StudyViewStatus.DownloadedJPGPreview -> {
                studyActivity.dicomGVAdapter.submitList(viewState.getListDicomItem())
            }
            StudyViewStatus.DownloadingMP4File -> {
                Log.w(TAG, "StudyViewStatus.DownloadingMP4File")
                studyActivity.openDownloadProgressDialog()
            }
            StudyViewStatus.DownloadedMP4File -> {
                studyActivity.closeDownloadProgressDialog()
            }
            StudyViewStatus.ExtractedMP4File -> {
                studyActivity.closeDownloadProgressDialog()
                Log.w(TAG, "StudyViewStatus.DownloadedMP4File number of frame: ${viewState.currentBitmap.size}")
                viewState.currentFileMP4Path?.let {
                    if (viewState.currentBitmap.size > 0)
                        studyActivity.showDicomPreviewDialog(it, viewState.currentBitmap)

                }

            }

        }
    }

    fun process(viewModel: StudyActVM, viewEvent: StudyViewEvent) {
        when (viewEvent) {

//            is StudyViewEvent.DownloadDicomPreview -> {
//                viewModel.viewStates().value?.let {
//                    viewModel.reduce(DownloadDicomPreviewJPGReducer(viewModel, viewState = it, viewEvent = viewEvent))
//                }
//            }

            is StudyViewEvent.DownloadJPGPreview -> {
                viewModel.viewStates().value?.let {
                    viewModel.reduce(DownloadJPGPreviewReducer(viewModel, viewState = it, viewEvent = viewEvent))
                }

            }

            is StudyViewEvent.GetInformationStudy -> {
                viewModel.viewStates().value?.let {
                    viewModel.reduce(GetInformationStudyReducer(viewModel, viewState = it, viewEvent = viewEvent))

                }
            }
            is StudyViewEvent.DownloadAndExtractMP4File -> {
                viewModel.viewStates().value?.let {
                    viewModel.reduce(DownloadMp4FileReducer(viewModel, viewState = it, viewEvent = viewEvent))

                }
            }

            is StudyViewEvent.ExtractMP4File -> {
                viewModel.viewStates().value?.let {
                    viewModel.reduce(ExtractMP4FileReducer(viewModel, viewState = it, viewEvent = viewEvent))

                }
            }
        }
    }

    inner class GetInformationStudyReducer(viewModel: StudyActVM, viewState: StudyViewState, val viewEvent: StudyViewEvent.GetInformationStudy)
        : StudyActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StudyStateEffectObject {
            viewModel.viewModelScope.launch {

                when (val result = studyScreenRepository.getInformationStudy(studyID = viewEvent.studyId)) {
                    is LCE.Result -> {
                        if (result.error) {
                            viewModel.reduce(
                                StudyStateEffectObject(
                                    viewState.copy(status = StudyViewStatus.LoadedStudyInformation), null
//                                    StudyViewEffect.ShowToast(message = "Loaded study information")
                                )
                            )

                        } else {
                            viewModel.reduce(
                                StudyStateEffectObject(
                                    viewState.copy(studyInformation=result.data, status = StudyViewStatus.LoadedStudyInformation, studyId = format("%06d", viewEvent.studyId)), null
//                                    StudyViewEffect.ShowToast(message = "Loaded study information")
                                )
                            )
                        }
                    }
                }


            }

            return StudyStateEffectObject()

        }

    }


    inner class DownloadJPGPreviewReducer(viewModel: StudyActVM, viewState: StudyViewState, val viewEvent: StudyViewEvent.DownloadJPGPreview)
        : StudyActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StudyStateEffectObject {
            viewModel.viewModelScope.launch {

                when (val result = studyScreenRepository.downloadJPGPreview(listener = listener, studyID = viewEvent.studyId, studyInstanceUID = viewEvent.studyInstanceUID)) {
                    is LCE.Result -> {

                        if (result.error) {
                            viewModel.reduce(StudyStateEffectObject(
                                viewState.copy(status = StudyViewStatus.DownloadedJPGPreview),
                                viewEffect = StudyViewEffect.ShowToast(message = "Error Downloading dicom preview")
                            ))
                        } else { // download success
                            // TODO process result.data
//                            Log.w(TAG, "${result.data}")
                            viewModel.reduce(StudyStateEffectObject(
                                viewState.copy(status = StudyViewStatus.DownloadedJPGPreview, studyPaths = result.data.filesPath), null
//                                viewEffect = StudyViewEffect.ShowToast(message = "Done Loading dicom preview")
                            ))
                        }
                    }
                }
            }

            return StudyStateEffectObject(
                viewState.copy(status = StudyViewStatus.DownloadingJPGPreview), null
//                viewEffect = StudyViewEffect.ShowToast(message = "On Loading dicom preview")
            )
        }

    }

    inner class DownloadMp4FileReducer(viewModel: StudyActVM, viewState: StudyViewState, val viewEvent: StudyViewEvent. DownloadAndExtractMP4File)
        : StudyActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StudyStateEffectObject {
            viewModel.viewModelScope.launch {

                when (val result = studyScreenRepository.downloadMP4File(listener = listener, studyID = viewEvent.studyId, studyInstanceUID = viewEvent.studyInstanceUID, relativePath = viewEvent.relativePath)) {
                    is LCE.Result -> {

                        if (result.error) {
                            viewModel.reduce(StudyStateEffectObject(
                                viewState.copy(status = StudyViewStatus.DownloadedMP4File),
                                viewEffect = StudyViewEffect.ShowToast(message = "Error Downloaded dicom mp4 preview")
                            ))
                        } else { // download success
                            // TODO process result.data
//                            Log.w(TAG, "Downloaded ${result.data}")
//                            val bitmaps = getFramesMP4FileDirect(result.data)
//                            viewModel.reduce(StudyStateEffectObject(
//                                viewState.copy(status = StudyViewStatus.DownloadedMP4File, currentFileMP4Path = result.data),
//                                viewEffect = StudyViewEffect.ShowToast(message = "Done Downloaded dicom mp4 preview ${result.data}")
//                            ))

                            viewModel.viewStates().value?.let {
                                viewModel.reduce(ExtractMP4FileReducer(viewModel, it, StudyViewEvent.ExtractMP4File(result.data)))
                            }

//                            viewModel.process()
                        }
                    }
                }
            }

            return StudyStateEffectObject(
                viewState.copy(status = StudyViewStatus.DownloadingMP4File), null
//                viewEffect = StudyViewEffect.ShowToast(message = "On Downloading dicom mp4 preview")
            )
        }

    }

    inner class ExtractMP4FileReducer(viewModel: StudyActVM, viewState: StudyViewState, val viewEvent: StudyViewEvent. ExtractMP4File)
        : StudyActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StudyStateEffectObject {
            viewModel.viewModelScope.launch {

                when (val result = getFramesMP4File(viewEvent.fileMP4Path)) {
                    is LCE.Result -> {

                        if (result.error) {
                            viewModel.reduce(StudyStateEffectObject(
                                viewState.copy(status = StudyViewStatus.ExtractedMP4File),
                                viewEffect = StudyViewEffect.ShowToast(message = "Error Extracted dicom mp4 preview")
                            ))
                        } else { // download success
                            // TODO process result.data
                            Log.w(TAG, "Downloaded ${result.data.size}")
                            viewModel.reduce(StudyStateEffectObject(
                                viewState.copy(status = StudyViewStatus.ExtractedMP4File, currentFileMP4Path = viewEvent.fileMP4Path, currentBitmap = result.data), null
//                                viewEffect = StudyViewEffect.ShowToast(message = "Done Extracted dicom mp4 preview ${result.data.size}")
                            ))
                        }
                    }
                }
            }

            return StudyStateEffectObject(
                viewState.copy(status = StudyViewStatus.ExtractingMP4File), null
//                viewEffect = StudyViewEffect.ShowToast(message = "On Extracting dicom mp4 preview")
            )
        }

    }

     fun getFramesMP4FileDirect(fileMP4Path: String): List<Bitmap>  {
        // TODO, extractor frame here and display

        val extrator = ExtractMPEGFrames()
//        val bitmaps = extrator.extractMPEGFrames(fileMP4Path) .map {
//            it.toGray()
//        }
         val bitmaps = extrator.extractMPEGFrames(fileMP4Path)
         return bitmaps
    }

    suspend fun getFramesMP4File(fileMP4Path: String): LCE< List<Bitmap> > = withContext(Dispatchers.IO) {
        // TODO, extractor frame here and display

        val extrator = ExtractMPEGFrames()
        val bitmaps = extrator.extractMPEGFrames(fileMP4Path)
//            .map {
//            it.toGray()
//        }

//        Log.w(DiskRepository.TAG, "getFramesMP4File ${fileMP4Path} ${bitmaps.size}")

        return@withContext LCE.Result(data = bitmaps, error = false, message = "no_error")
    }

}