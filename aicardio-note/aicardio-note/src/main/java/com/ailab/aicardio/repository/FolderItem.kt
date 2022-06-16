package com.ailab.aicardio.repository

import android.app.Activity
import android.content.Context

data class FolderItem(
    val description: String,
    val name: String,
    val path: String,
    val modifiedTime: Long,
    val isFile: Boolean,
    val hasAnnotation: Boolean
) {
    companion object {
        const val PREFS_NAME = "FOLDER_REMEMBER"
        const val KEY_FOLDER = "KEY_FOLDER"
        const val DEFAULT_FOLDER = "/storage/emulated/0/Download"

        fun getFolder(context: Context) : String? {
            val mPref = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return mPref.getString(KEY_FOLDER, DEFAULT_FOLDER)
        }

        fun remember(activity: Activity, folder: String){
//            Logger.getLogger(User.javaClass.name).warning("remember: $phone, $password")
            val editor = activity.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            editor.putString(KEY_FOLDER, folder)
            editor.apply()
        }
    }
}
