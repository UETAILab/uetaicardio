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

package com.ailab.aicardiotrainer.repositories

class SkillRepository {
    fun getSkillList(): List<SkillItem> {
        val skillList = listOf(
            SkillItem(
                "Measurement - EF",
                listOf(
                    StudyItem("Case 1", ""),
                    StudyItem("Case 2", ""),
                    StudyItem("Case 3", ""),
                    StudyItem("Case 4", ""),
                    StudyItem("Case 5", ""),
                    StudyItem("Case 6", ""),
                    StudyItem("Case 7", ""),
                    StudyItem("Case 8", ""),
                    StudyItem("Case 9", "")
                )
            ),
            SkillItem(
                "Measurement - GLS",
                listOf(
                    StudyItem("Case 1", ""),
                    StudyItem("Case 2", ""),
                    StudyItem("Case 3", ""),
                    StudyItem("Case 4", ""),
                    StudyItem("Case 5", ""),
                    StudyItem("Case 6", ""),
                    StudyItem("Case 7", ""),
                    StudyItem("Case 8", ""),
                    StudyItem("Case 9", "")
                )
            ),
            SkillItem(
                "Dianosis - EF",
                listOf(
                    StudyItem("Case 1", ""),
                    StudyItem("Case 2", ""),
                    StudyItem("Case 3", ""),
                    StudyItem("Case 4", ""),
                    StudyItem("Case 5", ""),
                    StudyItem("Case 6", ""),
                    StudyItem("Case 7", ""),
                    StudyItem("Case 8", ""),
                    StudyItem("Case 9", "")
                )
            ),
            SkillItem(
                "Dianosis - GLS",
                listOf(
                    StudyItem("Case 1", ""),
                    StudyItem("Case 2", ""),
                    StudyItem("Case 3", ""),
                    StudyItem("Case 4", ""),
                    StudyItem("Case 5", ""),
                    StudyItem("Case 6", ""),
                    StudyItem("Case 7", ""),
                    StudyItem("Case 8", ""),
                    StudyItem("Case 9", "")
                )
            ),
            SkillItem(
                "Mechanism",
                listOf(
                    StudyItem("Case 1", ""),
                    StudyItem("Case 2", ""),
                    StudyItem("Case 3", ""),
                    StudyItem("Case 4", ""),
                    StudyItem("Case 5", ""),
                    StudyItem("Case 6", ""),
                    StudyItem("Case 7", ""),
                    StudyItem("Case 8", ""),
                    StudyItem("Case 9", "")
                )
            ),
            SkillItem(
                "Prognosis",
                listOf(
                    StudyItem("Case 1", ""),
                    StudyItem("Case 2", ""),
                    StudyItem("Case 3", ""),
                    StudyItem("Case 4", ""),
                    StudyItem("Case 5", ""),
                    StudyItem("Case 6", ""),
                    StudyItem("Case 7", ""),
                    StudyItem("Case 8", ""),
                    StudyItem("Case 9", "")
                )
            )
        )

        return skillList
    }

    companion object {

        // For Singleton instantiation
        @Volatile
        private var instance: SkillRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: SkillRepository()
                        .also { instance = it }
            }
    }


}