require(tidyverse)
require(arrow)
require(skimr)
require(gganimate)
require(units)

base_dir <- "Big Data Bowl/"

#### somehow combine tracking files ####

tracking_files <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "weeks",full.names = T)) %>% 
  map(~list.files(.x,full.names = T))

schema1 <- schema(
  time = timestamp(unit = "ns",timezone = "GMT"),
  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),dir = float64(),
  event = string(),nflId = int64(),displayName = string(),jerseyNumber = int16(),position = string(),
  frameId = int16(),team = string(),gameId = int32(),playId = int32(),
  playDirection = string(),route = string()
)

schema2 <- schema(
  gameId = int32(),playId = int32(), nflId = int64(), frameId = int16(),
  time = timestamp(unit = "ns"),jerseyNumber = int16(),team = string(),playDirection = string(),
  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),
  dir = float64(),event = string()
)

schema3 <- schema(
  gameId = int32(),playId = int32(), nflId = int64(), displayName = string(), frameId = int16(),
  frametype = string(),time = timestamp(unit = "ns"),jerseyNumber = int16(),club = string(),
  playDirection = string(),
  x = float64(),y = float64(),s = float64(),a = float64(),dis = float64(),o = float64(),
  dir = float64(),event = string()
)

ds1 <- open_dataset(tracking_files[[1]],format = "csv",schema = schema1,skip = 1)
ds2 <- open_dataset(tracking_files[[2]],format = "csv",schema = schema2,skip = 1)
ds3 <- open_dataset(tracking_files[[3]],format = "csv",schema = schema3,skip = 1)

ds <- ds1 %>% 
  select(all_of(names(ds2))) %>% 
  select(-time) %>% 
  union_all(
    ds2 %>% 
      select(-time)
  ) %>% 
  union_all(
    ds3 %>% 
      rename(team = club) %>% 
      select(all_of(names(ds2))) %>% 
      select(-time)
  )

coverages_week1_2018 <- file.path(base_dir,"nfl-big-data-bowl-2021/coverages_week1.csv") %>% 
  read_csv

#### combine plays ####

plays <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "plays",full.names = T)) %>% 
  map(read_csv)

games <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "games",full.names = T)) %>% 
  map(read_csv)

games <- games[[1]] %>% 
  mutate(season = 2018) %>% 
  bind_rows(
    games[[2]],
    games[[3]] %>% 
      select(-homeFinalScore,visitorFinalScore)
  )

## fix the missing defensiveTeam field from 2018 data
plays[[1]] <- games %>% 
  pivot_longer(cols = c(homeTeamAbbr,visitorTeamAbbr)) %>% 
  group_by(gameId) %>% 
  summarise(across(c(value),list)) %>% 
  list(
    plays[[1]],
    .
  ) %>% 
  reduce(left_join,by = "gameId") %>% 
  mutate(defensiveTeam = pmap(list(value,possessionTeam),function(x,y) x[x != y])) %>% 
  unnest(cols = defensiveTeam) %>% 
  list(
    coverages_week1_2018 %>% 
      mutate(across(coverage,~sub(" \\w+$","",.x)),
             across(coverage,~sub(" ","-",.x))) %>% 
      rename(pff_passCoverage = coverage)
  ) %>% 
  reduce(inner_join,by = c("gameId","playId"))

## fix coverages from 2022 data
plays[[3]] <- plays[[3]] %>% 
  filter(isDropback == T) %>% 
  mutate(
    across(pff_passCoverage,~sub("Cover ","Cover-",.x)),
    across(pff_passCoverage,~ifelse(str_detect(.x,"Cover"),str_extract(.x,"Cover-\\d"),.x))
  )

plays_clean <- plays %>% 
  map(~select(.x,gameId,playId,playDescription,quarter,down,yardsToGo,possessionTeam,defensiveTeam,pff_passCoverage,
              preSnapHomeScore,preSnapVisitorScore,gameClock)) %>% 
  list(c(2018,2021,2022)) %>% 
  pmap(function(x,y) mutate(x,season = y)) %>% 
  bind_rows

#### get players ####

players <- base_dir %>% 
  list.files(pattern = "nfl-big-data-bowl",full.names = T) %>% 
  map(~list.files(.x,pattern = "players",full.names = T)) %>% 
  map(~read_csv(.x) %>% 
        setNames(c("nflId","height","weight","birthDate","collegeName","position","displayName"))) %>% 
  list(c(2018,2021,2022)) %>% 
  pmap(function(x,y) mutate(x,season = y)) %>% 
  bind_rows

#### set up graph parameters ####

theme_update(
  plot.caption = element_text(size = 12,vjust = .5,margin = margin(t = 40, b = 0)),
  plot.title = element_text(size = 16,hjust = .5),
  axis.text.x = element_blank()
)

color_grid <- read_csv("viz/color_grid.csv")

yard_lines <- function(yard_line) {
  
  ylabel = ifelse(yard_line > 0,yard_line,-yard_line)
  
  ylabel = ifelse(ylabel > 60, 110 - ylabel, ylabel - 10)
  
  return(ylabel)
  
}

field_lines <- list(
  geom_vline(xintercept = 29.71667,lty = 2),
  geom_vline(xintercept = 23.5833,lty = 2),
  geom_vline(xintercept = 12,lty = 3),
  geom_vline(xintercept = 6,lty = 3),
  geom_vline(xintercept = 53.33 - 12,lty = 3),
  geom_vline(xintercept = 53.33 - 6,lty = 3),
  geom_vline(xintercept = 0,lty = 1),
  geom_vline(xintercept = 53.33,lty = 1)
)

color_scheme <- plays_clean %>% 
  left_join(
    color_grid %>% 
      mutate(across(c(team1,team2),~ifelse(. == "LAR","LA",.))) %>% 
      select(team1,team2,possessionColor = color1,defensiveColor = color2),
    by = c("possessionTeam" = "team1","defensiveTeam" = "team2")
  ) %>% 
  list(
    pivot_longer(.,cols = matches("Team"),names_to = "side",values_to = "team") %>% 
      select(gameId,playId,side,team) %>% 
      mutate(across(side,~sub("Team","",.))),
    pivot_longer(.,cols = matches("color"),names_to = "side",values_to = "color") %>% 
      select(gameId,playId,side,color) %>% 
      mutate(across(side,~sub("Color","",.)))
  ) %>% 
  .[-1] %>% 
  reduce(left_join,by = c("gameId","playId","side")) %>% 
  select(-side)

play_details <- plays_clean %>% 
  select(-season) %>% 
  left_join(
    games,
    by = "gameId"
  ) %>% 
  mutate(across(c(possessionTeam,defensiveTeam),list(score = ~ifelse(.x == homeTeamAbbr,preSnapHomeScore,preSnapVisitorScore)))) %>% 
  mutate(
    down_distance = paste(scales::ordinal(down),"&",yardsToGo),
    scoreboard = paste(possessionTeam_score,defensiveTeam_score,sep = " - "),
    sitch = paste(scales::ordinal(quarter),"Quarter,",str_extract(gameClock,"\\d+:\\d+")),
    location = ifelse(possessionTeam == homeTeamAbbr,"v","@"),
    across(location,~paste0(possessionTeam," ",.x," ",defensiveTeam," ",gameDate))
  ) %>% 
  rowwise %>% 
  mutate(caption = paste(location,sitch,scoreboard,down_distance,sep = "\n")) %>% 
  ungroup

play_animator <- function(ds, game_id, play_id, play_details, color_scheme, field_lines) {
  
  play_frames <- ds %>% 
    filter(playId == play_id,gameId == game_id) %>% 
    collect %>% 
    list(
      filter(.,event == "ball_snap") %>% 
        mutate(across(frameId,~.x - 1)) %>% 
        select(frameId,gameId,playId)
    ) %>% 
    magrittr::inset2(
      length(.) + 1,
      semi_join(
        .[[1]] %>% 
          filter(team == "football"),
        .[[2]],
        by = c("gameId","playId","frameId")
      ) %>% 
        select(spot = x)
    ) %>% 
    .[-2] %>% 
    bind_cols %>% 
    mutate_at(vars(jerseyNumber),~ifelse(team == "football",NA,as.integer(.))) %>% 
    mutate_at(vars(o),~replace_na(.,0)) %>% 
    group_by(nflId) %>%  
    mutate(frame_level = factor(frameId)) %>% 
    ungroup
  
  play_caption <- play_details %>% 
    semi_join(play_selection,by = c("playId","gameId")) %>% 
    pull(caption)
  
  play_df <- play_frames %>% 
    left_join(
      color_scheme,
      by = c("gameId","playId","team")
    ) %>% 
    left_join(
      play_details,
      by = c("gameId","playId")
    ) %>% 
    left_join(
      players %>% 
        select(nflId,position,season),
      by = c("nflId","season")
    ) %>% 
    mutate(skill = ifelse(position %in% c("WR","FB","TE","RB"),"black",NA)) %>% 
    #filter(event == "ball_snap") %>% #print(width = Inf)
    mutate(
      flip_direction = ifelse(playDirection == "right","counterclockwise","clockwise"),
      newx = ifelse(flip_direction == "counterclockwise", y, -y + 53.33),
      newy = ifelse(flip_direction == "counterclockwise", -x, x),
      newo = ifelse(flip_direction == "counterclockwise",-(o +90), -(o - 90)),
      newspot = ifelse(flip_direction == "counterclockwise", -spot,spot),
      line_to_gain = newspot - yardsToGo,
      across(color,~ifelse(team == "football","#8d4d37",.x)),
      number_color = ifelse(color == "#FFFFFF","black","white"),
      shape = ifelse(team == "football",18,19)
    )
  
  yard_line_range <- play_df$newy %>% 
    range
  
  goal_lines <- list()
  
  for (spot in c(10,110,-10,-110)) {
    
    if (between(spot,yard_line_range[1],yard_line_range[2])) {
      
      goal_line <- geom_hline(yintercept = spot,color = "red2")
      
    } else {
      
      goal_line <- list()
      
    }
    
    goal_lines <- goal_lines %>% 
      magrittr::inset2(
        length(.) + 1,
        goal_line
      )
    
  }
  
  route_lines <- play_df %>% 
    bind_cols(
      filter(.,event == "ball_snap") %>% 
        distinct(frameId) %>% 
        rename(ball_snap = frameId)
    ) %>% 
    filter(skill == "black",frameId >= ball_snap) %>% 
    select(-frame_level)
  
  final_plot <- play_df %>% 
    {
      ggplot(.,mapping = aes(x = newx, y = newy))+
        field_lines +
        geom_hline(yintercept = unique(.$newspot),color = "navy")+
        geom_hline(yintercept = unique(.$line_to_gain),color = "yellow2",lwd = 2)+
        goal_lines +
        geom_path(data = route_lines, aes(x = newx, y = newy,group = nflId), color = "black")+
        geom_point(aes(color = color,shape = shape),size = 7,show.legend = F)+
        scale_color_identity()+
        scale_shape_identity()+
        geom_text(aes(label = jerseyNumber,angle = newo,color = number_color))+
        scale_y_continuous(labels = yard_lines)+
        labs(x = "",y = "",caption = play_caption)+
        ggtitle(label = unique(.$pff_passCoverage))+
        coord_fixed(ratio = 1) +
        transition_states(frame_level,transition_length = 2,state_length = 1)
    }
    
  animate(final_plot,fps = 5)
  
}

play_selection <- plays_clean %>% 
  filter(str_detect(gameId,"2022")) %>% 
  distinct(playId,gameId) %>% 
  collect %>% 
  slice_sample(n = 1)

play_animator(ds, play_selection$gameId, play_selection$playId, play_details, color_scheme, field_lines)
