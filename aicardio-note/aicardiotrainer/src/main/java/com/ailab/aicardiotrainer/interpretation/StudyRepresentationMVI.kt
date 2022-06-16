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

import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.studyscreen.*
import com.ailab.aicardiotrainer.studyscreen.ExtractMPEGFrames
import kotlinx.android.synthetic.main.activity_interpretation.*
import kotlinx.android.synthetic.main.activity_study.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class StudyRepresentationMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: StudyRepresentationMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: StudyRepresentationMVI()
                        .also { instance = it }
            }

        const val TAG = "StudyRepresentationMVI"

        fun process(InterpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(InterpretationActVM, InterpretationViewEvent)
        }

        fun renderViewState(InterpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
            getInstance().renderViewState(InterpretationActivity, viewState)
        }

        fun renderViewEffect(InterpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(InterpretationActivity, viewEffect)
        }
    }
    private val diskRepository = DiskRepository.getInstance()

    private fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
    }

    private fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
        Log.w(TAG, "renderViewStateMVI ${viewState.status.javaClass.name}")
        when(viewState.status) {

            InterpretationViewStatus.OnDoneLoadingRepresentationStudyInstanceUID -> {
                interpretationActivity.studyRepresentationGVAdapter.submitList(viewState.getListSopInstanceUIDItem())
            }

            InterpretationViewStatus.OnLoadedMP4File -> {
//                interpretationActivity.studyRepresentationGVAdapter.submitList(viewState.getListSopInstanceUIDItem())

                interpretationActivity.interpretationFrameRVAdapter.submitList(viewState.getSubmitListFrameItem())

                interpretationActivity.viewModel.getRenderMP4FrameObject()?.let {
                    InterpretationPlaybackMVI.renderViewEffect(interpretationActivity, InterpretationViewEffect.RenderMP4Frame(it))
                }


            }
        }
    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {

            is InterpretationViewEvent.LoadingRepresentationStudyInstanceUID -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(LoadingStudyRepresentationReducerAsync(interpretationActVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.PlayBackMP4File -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(PlayBackMp4FileReducerAsync(interpretationActVM, it, interpretationViewEvent))
                }
            }

        }
    }

    inner class Reducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StateEffectObject {
            when(viewEvent) {

            }
            return StateEffectObject()
        }

    }


    inner class PlayBackMp4FileReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.PlayBackMP4File) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            if (viewState.status is InterpretationViewStatus.OnLoadingMP4File) return StateEffectObject()

            val result = StateEffectObject(viewState = viewState.copy(status = InterpretationViewStatus.OnLoadingMP4File))

            viewModel.viewModelScope.launch {

                when (val resultLaunch = getFramesMP4File(viewEvent.fileMP4Path)) {
                    is LCE.Result -> {

                        if (resultLaunch.error) {
                            viewModel.reduce(StateEffectObject(
                                viewState.copy(status = InterpretationViewStatus.OnLoadedMP4File),
                                viewEffect = InterpretationViewEffect.ShowToast(message = "Error Extracted dicom mp4 preview")
                            ))
                        } else { // download success
                            // TODO process result.data
                            viewModel.viewStates().value?.let {
                                viewModel.reduce(StateEffectObject(
                                    it.copy(status = InterpretationViewStatus.OnLoadedMP4File,
                                        sopInstanceUIDPath = viewEvent.fileMP4Path, dicomAnnotation = DicomAnnotation.getNewAnnotation(resultLaunch.data.size),
                                        machineAnnotation =  DicomAnnotation.getNewAnnotation(resultLaunch.data.size),
                                        sopInstanceUIDBitmaps = resultLaunch.data), null
//                                viewEffect = StudyViewEffect.ShowToast(message = "Done Extracted dicom mp4 preview ${result.data.size}")
                                ))
                            }

                        }
                    }
                }
            }
            return result
        }
    }
    suspend fun getFramesMP4File(fileMP4Path: String): LCE<List<Bitmap>> = withContext(Dispatchers.IO) {
        // TODO, extractor frame here and display

        val extrator = ExtractMPEGFrames()
        val bitmaps = extrator.extractMPEGFrames(fileMP4Path)
        return@withContext LCE.Result(data = bitmaps, error = false, message = "no_error")
    }

//    inner class DownloadMp4FileReducer(viewModel: StudyActVM, viewState: StudyViewState, val viewEvent: StudyViewEvent. DownloadAndExtractMP4File)
//        : StudyActReducer(viewModel, viewState, viewEvent) {
//
//        override fun reduce(): StudyStateEffectObject {
//            viewModel.viewModelScope.launch {
//
//                when (val result = studyScreenRepository.downloadMP4File(listener = listener, studyID = viewEvent.studyId, studyInstanceUID = viewEvent.studyInstanceUID, relativePath = viewEvent.relativePath)) {
//                    is LCE.Result -> {
//
//                        if (result.error) {
//                            viewModel.reduce(StudyStateEffectObject(
//                                viewState.copy(status = StudyViewStatus.DownloadedMP4File),
//                                viewEffect = StudyViewEffect.ShowToast(message = "Error Downloaded dicom mp4 preview")
//                            ))
//                        } else { // download success
//                            // TODO process result.data
////                            Log.w(TAG, "Downloaded ${result.data}")
////                            val bitmaps = getFramesMP4FileDirect(result.data)
////                            viewModel.reduce(StudyStateEffectObject(
////                                viewState.copy(status = StudyViewStatus.DownloadedMP4File, currentFileMP4Path = result.data),
////                                viewEffect = StudyViewEffect.ShowToast(message = "Done Downloaded dicom mp4 preview ${result.data}")
////                            ))
//
//                            viewModel.viewStates().value?.let {
//                                viewModel.reduce(ExtractMP4FileReducer(viewModel, it, StudyViewEvent.ExtractMP4File(result.data)))
//                            }
//
////                            viewModel.process()
//                        }
//                    }
//                }
//            }
//
//            return StudyStateEffectObject(
//                viewState.copy(status = StudyViewStatus.DownloadingMP4File), null
////                viewEffect = StudyViewEffect.ShowToast(message = "On Downloading dicom mp4 preview")
//            )
//        }
//
//    }

    inner class LoadingStudyRepresentationReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.LoadingRepresentationStudyInstanceUID) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            if (viewState.status is InterpretationViewStatus.OnLoadingRepresentationStudyInstanceUID) return StateEffectObject()
            val result = StateEffectObject(viewState = viewState.copy(status = InterpretationViewStatus.OnLoadingRepresentationStudyInstanceUID))

            viewModel.viewModelScope.launch {

                val resultLaunch = diskRepository.getRepresentationInStudyInstanceUID(viewEvent.studyInstanceUID)
                Log.w(TAG, "resultLaunch: ${resultLaunch}")
                if (resultLaunch.error == false) {
                    // process
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(
                            StateEffectObject(
                                viewState = it.copy(status = InterpretationViewStatus.OnDoneLoadingRepresentationStudyInstanceUID,
                                    studyFiles = resultLaunch.data.representationFiles,
                                    studyRepresentation = resultLaunch.data.representationBitmap),
                                viewEffect = InterpretationViewEffect.ShowToast("DONE Loading Representation StudyInstanceUID")
                            )
                        )
                    }

                } else {
                    // error when loading presentatinon
                    viewModel.reduce(
                        StateEffectObject(
                            null,
                            viewEffect = InterpretationViewEffect.ShowToast("ERROR Loading Representation StudyInstanceUID")
                        )
                    )
                }

            }

            return result
        }

    }
}