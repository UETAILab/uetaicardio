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

package com.ailab.aicardiotrainer.mainscreen

import android.app.Application
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardiotrainer.repositories.StudyRepository
import com.rohitss.aacmvi.AacMviViewModel
import kotlinx.coroutines.launch

class TrainerActVM(application: Application) : AacMviViewModel<TrainerState, TrainerViewEffect, TrainerViewEvent>(application) {
    companion object {
        const val TAG = "TrainerActVM"
    }
    init {
        viewState = TrainerState(
            status = TrainerViewStatus.Start
        )
    }

    override fun process(viewEvent: TrainerViewEvent) {
        super.process(viewEvent)
//        when (viewEvent) {
//            is TrainerViewEvent.DownloadStudy -> downloadStudy(viewEvent)
//        }
    }

//    private fun downloadStudy(viewEvent: TrainerViewEvent.DownloadStudy) {
//        if (viewState.status == TrainerViewStatus.Downloading) {
//            Log.w(TAG, "downloadStudy: Downloading is in progress")
//            return
//        }
//
//        viewState = viewState.copy(status = TrainerViewStatus.Downloading)
//        viewModelScope.launch {
//            val skillName = viewEvent.skillName
//            val study = viewEvent.study
//            StudyRepository.getInstance().getFileDicom(skillName, study)
//            viewState = viewState.copy(status = TrainerViewStatus.Downloaded)
//        }
//    }
}
