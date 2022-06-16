package com.ailab.aicardio.mainscreen

import com.ailab.aicardio.repository.FolderItem


data class MainViewState(
    val mainViewStatus: MainViewStatus,
    val folderList: List<FolderItem>,
    val permissionGranted: Boolean,
    val folder: String
)

sealed class MainViewEffect {
    data class ShowSnackbar(val message: String) : MainViewEffect()
    data class ShowToast(val message: String) : MainViewEffect()
}

sealed class MainViewEvent {
    data class NewsItemClicked(val folderItem: FolderItem, val activity: MainActivity) : MainViewEvent()
    data class NewsItemLongClicked(val folderItem: FolderItem, val activity: MainActivity) : MainViewEvent()
    object FabClicked : MainViewEvent()
    object OnSwipeRefresh : MainViewEvent()
    data class FetchNews(val folder: String = "", val permissionGranted: Boolean) : MainViewEvent()
    object PermissionGranted : MainViewEvent()
    object SortByName : MainViewEvent()
    object SortByTime : MainViewEvent()
}

sealed class MainViewStatus {
    object Fetching : MainViewStatus()
    object Fetched : MainViewStatus()
    object NotFetched : MainViewStatus()
    data class OpenAnnotateActivity(val folder: String, val file: String?) : MainViewStatus()
    data class OpenMainActivity(val folder: String) : MainViewStatus()
}