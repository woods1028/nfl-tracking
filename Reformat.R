require(tidyverse)
require(arrow)
require(skimr)
require(gganimate)
require(units)

base_dir <- "~/Dropbox/NFL/Big Data Bowl/"

#### somehow combine tracking files ####

tracking_files <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "weeks",full.names = T)) %>% 
  map(~list.files(.x,full.names = T))

schema1 <- schema(time = timestamp(unit = "ns",timezone = "GMT"),
                  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),dir = float64(),
                  event = string(),nflId = int64(),displayName = string(),jerseyNumber = int16(),position = string(),
                  frameId = int16(),team = string(),gameId = int32(),playId = int32(),
                  playDirection = string(),route = string())

schema2 <- schema(gameId = int32(),playId = int32(), nflId = int64(), frameId = int16(),
                  time = timestamp(unit = "ns"),jerseyNumber = int16(),team = string(),playDirection = string(),
                  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),
                  dir = float64(),event = string())

schema3 <- schema(gameId = int32(),playId = int32(), nflId = int64(), displayName = string(), frameId = int16(),
                  frametype = string(),time = timestamp(unit = "ns"),jerseyNumber = int16(),club = string(),
                  playDirection = string(),
                  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),
                  dir = float64(),event = string())

ds1 <- open_dataset(tracking_files[[1]],format = "csv",schema = schema1,skip = 1)
ds2 <- open_dataset(tracking_files[[2]],format = "csv",schema = schema2,skip = 1)
ds3 <- open_dataset(tracking_files[[3]],format = "csv",schema = schema3,skip = 1)

plays <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "plays",full.names = T)) %>% 
  map(read_csv)

dest_dir <- "reformatted"

ds3 %>% 
  list(
    filter(.,event %in% c("ball_snap","autoevent_ballsnap")) %>% 
      group_by(gameId,playId) %>% 
      summarise(snap = min(frameId),
                .groups = "drop")
  ) %>% 
  reduce(inner_join,by = c("gameId","playId")) %>% 
  filter(frameId >= snap,
         frameId <= snap + 50) %>% 
  rename(team = club) %>% 
  select(all_of(names(ds2))) %>% 
  semi_join(
    plays[[3]] %>% 
      filter(isDropback == T) %>% 
      select(gameId,playId) %>% 
      mutate(across(c(gameId,playId),as.integer)),
    on = c("gameId","playId")
  ) %>% 
  mutate(across(event,~ifelse(. == "NA",NA,.))) %>% 
  write_dataset(
    path = "reformatted/weeks22",
    basename_template = paste0("part-{i}-22.", as.character("parquet")),
    format = "parquet",
    max_rows_per_file = 9e5
  )

ds2 %>% 
  list(
    filter(.,event %in% c("ball_snap","autoevent_ballsnap")) %>% 
      group_by(gameId,playId) %>% 
      summarise(snap = min(frameId),
                .groups = "drop")
  ) %>% 
  reduce(inner_join,by = c("gameId","playId")) %>% 
  filter(frameId >= snap,
         frameId <= snap + 50) %>% 
  mutate(across(event,~ifelse(. == "NA",NA,.))) %>% 
  write_dataset(
    path = "reformatted/weeks21",
    basename_template = paste0("part-{i}-21.", as.character("parquet")),
    format = "parquet",
    max_rows_per_file = 9e5
  )

#### plays ####

plays_colnames <- "
  gameId number,
  playId number,
  playDescription varchar(540),
  quarter number,
  down number,
  yardsToGo number,
  possessionTeam varchar(3),
  defensiveTeam varchar(3),
  yardlineSide varchar(4),
  yardlineNumber number,
  gameClock varchar(10),
  preSnapHomeScore number,
  preSnapVisitorScore number,
  passResult varchar(2),
  penaltyYards varchar(3),
  absoluteYardlineNumber varchar(3),
  offenseFormation varchar(10),
  pff_passCoverage varchar(13)
" %>% 
  str_split("\n") %>% 
  .[[1]] %>% 
  str_extract("\\w+") %>% 
  na.omit

plays[[3]] %>% 
  filter(isDropback == T) %>% 
  mutate(
    across(pff_passCoverage,~sub("Cover ","Cover-",.x)),
    across(pff_passCoverage,~ifelse(str_detect(.x,"Cover"),str_extract(.x,"Cover-\\d"),.x))
  ) %>% 
  select(all_of(plays_colnames)) %>% 
  write_dataset(
    path = "reformatted/plays22",
    basename_template = paste0("part-{i}-22.", as.character("parquet")),
    format = "parquet",
    max_rows_per_file = 9e5
  )

plays[[2]] %>% 
  select(all_of(plays_colnames)) %>% 
  write_dataset(
    path = "reformatted/plays21",
    basename_template = paste0("part-{i}-21.", as.character("parquet")),
    format = "parquet",
    max_rows_per_file = 9e5
  )
