package com.ailab.aicardio.repository

import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.ailab.aicardio.LCE
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.*

@RequiresApi(Build.VERSION_CODES.N)
class FolderRepository {


    companion object {

        val DEFAULT_FOLDER_DOWNLOAD = "/storage/emulated/0/Download"
//        val DEFAULT_FOLDER_DOWNLOAD = "/storage/emulated/0/Android/data/com.ailab.aicardio/files/Download"

        val SORT_TYPE_NAME_DESC = "name_desc"
        val SORT_TYPE_TIME_DESC = "time_desc"
        val SORT_TYPE_NAME_ASC = "name_asc"
        val SORT_TYPE_TIME_ASC = "time_asc"

        // For Singleton instantiation
        @Volatile
        private var instance: FolderRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: FolderRepository()
                        .also { instance = it }
            }

        fun getIsAnnotatedFile(filePath: String) : Boolean {
            return File(filePath + ".json").exists()
        }
    }





    suspend fun getSetLatestFolderList(folderPath: String, type: String = SORT_TYPE_NAME_ASC): LCE<List<FolderItem>> = withContext(Dispatchers.IO) {

        val folder = File(folderPath)

        val folderList = sortByKey(
            folder.listFiles()?.filter{
                !it.absolutePath.contains(".thumbnail") && !it.absolutePath.contains(".json")
            }?.map {
                FolderItem(description = it.absolutePath, name = it.absolutePath, path = it.absolutePath,
                    modifiedTime = it.lastModified(), isFile = it.isFile, hasAnnotation = File(it.absolutePath+".json").exists())
            } ?: emptyList(),
            type = type
        )

        Log.w("CHECK DATA", "$folderList ${folder.listFiles()}")

        return@withContext if (folderList.isNotEmpty()) {
            LCE.Result(data = folderList, error = false, message = "no error")
        } else {
            LCE.Result(data = emptyList(), error = true, message = "error")
        }
    }

    fun sortByKey(folderList: List<FolderItem>, type: String=SORT_TYPE_NAME_ASC): List<FolderItem> {
        // key ["name", "time"]
        when (type) {
            SORT_TYPE_NAME_DESC -> {
                return folderList.sortedWith(
                    compareBy<FolderItem>({ it.name }, { it.modifiedTime }).reversed()
                )
            }
            SORT_TYPE_NAME_ASC -> {
                return folderList.sortedWith(
                    compareBy<FolderItem>({ it.name }, { it.modifiedTime })
                )
            }

            SORT_TYPE_TIME_DESC -> {
                return folderList.sortedWith(
                    compareBy<FolderItem>({ it.modifiedTime }, { it.name }).reversed()
                )
            }
            SORT_TYPE_TIME_ASC -> {
                return folderList.sortedWith(
                    compareBy<FolderItem>({ it.modifiedTime }, { it.name })
                )
            }
        }
        return folderList
    }

}