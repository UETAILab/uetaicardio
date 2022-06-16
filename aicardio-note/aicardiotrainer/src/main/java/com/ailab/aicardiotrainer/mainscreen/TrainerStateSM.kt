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

import com.ailab.aicardiotrainer.repositories.StudyItem

data class TrainerState(
    val status: TrainerViewStatus
)
{

}

sealed class TrainerViewEffect {

}

sealed class TrainerViewEvent {
    data class DownloadStudy(val skillName: String, val study: StudyItem) : TrainerViewEvent()
}

sealed class TrainerViewStatus {
    object Start : TrainerViewStatus()
    object Downloading : TrainerViewStatus()
    object Downloaded : TrainerViewStatus()
}