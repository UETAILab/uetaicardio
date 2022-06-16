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

import android.graphics.Bitmap
import com.imebra.DataSet
import org.json.JSONObject

data class DicomObject(
    val dicomPath: String = "",
    val dataset: DataSet = DataSet(),
    val tags: JSONObject = JSONObject(),
    val bitmaps: List<Bitmap> = emptyList()
)