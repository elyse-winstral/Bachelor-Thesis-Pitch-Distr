library(baseballr)
library(progress)

year <- 2023
dfs <- list()
temp_game_pks =mlb_schedule(year, level_ids="1")
game_pks <- subset(temp_game_pks, temp_game_pks$game_type == "R")$game_pk
game_pks <- unique(game_pks)
# game_pks should be 162*15 (amount of total games per regular season)

game_pks_1 <- game_pks[1: 900]
game_pks_2 <- game_pks[!game_pks %in% game_pks_1] #all remaining games in game_pks

# copy_games <- c(game_pks_1, game_pks_2) #has 2430 elts
# diffs <- game_pks[!game_pks %in% copy_games] #is empty

# displays progress bar
pbar=progress_bar$new(total=length(game_pks_2))
#
# CHANGE GAME_PKS_1 TO 2
for (game in game_pks_2) {
    #cat(game, sep="\n")
    temp_df <- mlb_batting_orders(game)
    temp_df$game_pk <- game
    temp_df <- temp_df[, c('id', 'batting_order', 'game_pk', 'fullName')]
    dfs[[length(dfs) + 1]] <- temp_df
    pbar$tick()
}
df <- do.call(rbind, dfs)

file_name <- sprintf("2_batting_order_%d.csv", year)
write.csv(df, file = file_name, row.names = FALSE)
