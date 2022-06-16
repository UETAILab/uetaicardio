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
import android.util.Log
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.interfaces.LoadingProgressListener
import com.imebra.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.nio.ByteBuffer

class DicomRepository {
    companion object {

        // For Singleton instantiation
        @Volatile
        private var instance: DicomRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: DicomRepository()
                        .also { instance = it }
            }
        const val TAG = "DicomRepository"
    }

    suspend fun getDatasetAndBitmaps(
        dicomFile: String,
        listener: LoadingProgressListener
    ): LCE<DicomObject>
            = withContext(Dispatchers.IO) {
        val result = getDicomObjectFromFile(dicomFile, listener)

        if (result.bitmaps.isNotEmpty())
            LCE.Result(data = result, error = false, message = "dicom read")
        else
            LCE.Result(data = result, error = true, message = "dicom ${dicomFile} cannot read")
    }

    suspend fun getDicomObjectFromFile(
        file_name: String,
        listener: LoadingProgressListener
    ): DicomObject = withContext(Dispatchers.IO) {

        val arrayListBitmap = ArrayList<Bitmap>()


        try {
            val loadedDataSet = com.imebra.CodecFactory.load(file_name)
            loadedDataSet.getImageApplyModalityTransform(0)
//            loadedDataSet.
            var image: com.imebra.Image
            var frameNumber: Long = 0

//            val chain = TransformsChain(img)
//            val bitmap = getBitmap

            val luts = ArrayList<LUT>()
            var scanLUTs: Long = 0
            while (true) {
                try {
                    luts.add(
                        loadedDataSet.getLUT(
                            TagId(0x0028, 0x3010),
                            scanLUTs
                        )
                    )
                } catch (e: Exception) {
                    e.printStackTrace()
                    break
                }

                scanLUTs++
            }
            val vois = loadedDataSet.voIs
            image = loadedDataSet.getImage(0)
//                 Get the size in pixels
            val width = image.width
            val height = image.height

            val chain = TransformsChain()

            if (ColorTransformsFactory.isMonochrome(image.colorSpace)) {
                val voilutTransform = VOILUT()
                // Retrieve the VOIs (center/width pairs)
                if (!vois.isEmpty) {
                    voilutTransform.setCenterWidth(vois.get(0).center, vois.get(0).width)
                } else if (luts.isNotEmpty()) {
                    voilutTransform.setLUT(luts[0])
                } else {
                    voilutTransform.applyOptimalVOI(image, 0, 0, width, height)
                }

                chain.addTransform(voilutTransform)
            }

            val draw = DrawBitmap(chain)

            while (true) {
                try {
                    image = loadedDataSet.getImage(frameNumber)
                    val requestedBufferSize = draw.getBitmap(image, drawBitmapType_t.drawBitmapRGBA, 4, ByteArray(0))

                    val buffer = ByteArray(requestedBufferSize.toInt()) // Ideally you want to reuse this in subsequent calls to getBitmap()

                    val byteBuffer = ByteBuffer.wrap(buffer)

                    draw.getBitmap(image, drawBitmapType_t.drawBitmapRGBA, 4, buffer)
                    val renderBitmap =
                        Bitmap.createBitmap(image.width.toInt(), image.height.toInt(), Bitmap.Config.ARGB_8888)
                    renderBitmap.copyPixelsFromBuffer(byteBuffer)
                    arrayListBitmap.add(renderBitmap)
                    frameNumber += 1
                    listener.onProgress(frameNumber)
                } catch (e: Exception) {
                    break
                }
            }

            val tagsValue = getTagsDicomValue(loadedDataSet)
            return@withContext DicomObject(file_name, loadedDataSet, tagsValue, arrayListBitmap)

        } catch (e : Exception) {
            return@withContext DicomObject(dicomPath = file_name, dataset = DataSet(), tags = JSONObject(), bitmaps = emptyList()  )
        }
    }

    fun getTagsDicomValue(dataSet: DataSet): JSONObject {
        val details = JSONObject()
        try {
            val tags = dataSet.tags


            val n = tags.size().toInt()

            for (i in 0 until n) {
                val tag = tags.get(i)
                val typeData = dataSet.getDataType(tag)
                Log.w(TAG, "${tag} -- ${typeData}")
//                imebraJNI.
//                val sequence = imebraJNI.Tag_sequenceItemExists()
//                DicomDictionary.
//                dataSet.getSequencxeItem(tag, )

                if (tag.groupId == 0x7FE0) continue // {Format TagGroupId: 7FE0, 7FE1} (pixel_data, private_creator)
//                tag.
                try {
                    val tagValue = dataSet.getString(tag, 0)

                    val tagName = DicomDictionary.getTagName(tag).toString()
                    val tagId = "(${String.format("%04X",tag.groupId)},${String.format("%04X",tag.tagId)})"
                    details.put(tagId, tagValue)
                    Log.w(TAG, "$tagName $tagValue $tagId")
                    Log.w("TAG", "TUAN01 ${DicomDictionary.getUnicodeTagName(tag)} TUAN02 ${DicomDictionary.getTagType(tag)}")

                } catch (e: Exception) {
                    Log.w("Exception", "${e}")
                }

            }

        } catch (e: Exception) {
            Log.w(TAG, "getTagsDicomValue ${e}")

        }
        return details
    }


    suspend fun getTagDicom(loadedDataSet: DataSet) : LCE<JSONObject>  = withContext(Dispatchers.IO){
        val details = JSONObject()

        val tags = loadedDataSet.tags
        val n = tags.size().toInt()

        for (i in 0 until n) {
            val tag = tags.get(i)

            if (tag.groupId == 0x7FE0) continue // {Format TagGroupId: 7FE0, 7FE1} (pixel_data, private_creator)

            try {
                val tagValue = loadedDataSet.getString(tag, 0)
                val tagName = DicomDictionary.getTagName(tag).toString()
                val tagId = "(${String.format("%04X",tag.groupId)},${String.format("%04X",tag.tagId)})"
                details.put(tagId, tagValue)
                Log.w(TAG, "$tagName $tagValue $tagId")

            } catch (e: Exception) {

            }

        }
        return@withContext LCE.Result(data = details, error = false, message = "dicom read Tag Value")
    }

}