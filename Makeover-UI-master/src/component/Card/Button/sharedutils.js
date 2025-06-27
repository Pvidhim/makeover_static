export const getShadesAndEndpoint = (type) => {
  if (!["lipstick", "lip_liner", "eyeliner", "eyeshadow", "concealer", "eyebrows", "hair"].includes(type)) {
  throw new Error("Invalid parameter: type should be 'lipstick', 'lip_liner', 'eyeliner', 'eyeshadow', 'concealer', 'eyebrows', or 'hair'.");
}


  const concealerShades = [
    { name: "C_01_Porcelain", color: "#F8E4DB" },        // Fair skin with neutral undertone
    { name: "C_02_FairBeige", color: "#F3D1B8" },        // Light skin with warm undertone
    { name: "C_03_NudeIvory", color: "#E6BEA4" },        // Light-medium skin with neutral undertone
    { name: "C_04_GoldenSand", color: "#D2A074" },       // Medium skin with golden undertone
    { name: "C_05_WarmHoney", color: "#B88463" },        // Tan skin with warm undertone
    { name: "C_06_AlmondSpice", color: "#9D6D50" },      // Medium-dark skin with neutral undertone
    { name: "C_07_CaramelBronze", color: "#7C5138" },    // Dark tan skin with golden undertone
    { name: "C_08_MochaChestnut", color: "#5B3A29" },    // Deep skin with warm undertone
];


  const eyebrowShades = [
    { name: "EB_01_3B2F2F", color: "#3B2F2F" },
    { name: "EB_02_4E3B31", color: "#4E3B31" },
    { name: "EB_03_5A4B3C", color: "#5A4B3C" },
    { name: "EB_04_6E5848", color: "#6E5848" },
    { name: "EB_05_8A6E5B", color: "#8A6E5B" },
  ];

  const matteShades = [
    { name: "M_01", color: "#700415", subcategory: "kiss_sensation" },
    { name: "M_02", color: "#81071C" , subcategory: "kiss_sensation"},
    { name: "M_03", color: "#8F131C" , subcategory: "kiss_sensation"},
    { name: "M_04", color: "#831E26" , subcategory: "kiss_sensation"},
    { name: "M_05", color: "#66161A" , subcategory: "kiss_sensation"},
    { name: "M_06", color: "#3C1213" , subcategory: "kiss_sensation"},
    { name: "M_07", color: "#401310" , subcategory: "kiss_sensation"},
    { name: "M_08", color: "#6E182E" , subcategory: "kiss_sensation"},
    { name: "M_09", color: "#801620" , subcategory: "kiss_sensation"},
    { name: "M_10", color: "#A95B57" , subcategory: "kiss_sensation"},
    { name: "M_11", color: "#5D0D10" , subcategory: "kiss_sensation"},
    { name: "M_12", color: "#983445" , subcategory: "kiss_sensation"},
    { name: "M_13", color: "#60262F" , subcategory: "kiss_sensation"},
    { name: "M_14", color: "#6E263D" , subcategory: "kiss_sensation"},
    { name: "M_15", color: "#812937" , subcategory: "kiss_sensation"},
    { name: "M_16", color: "#BF4951" , subcategory: "kiss_sensation"},
    { name: "M_17", color: "#5F2E23" , subcategory: "kiss_sensation"},
    { name: "M_18", color: "#481F1B" , subcategory: "kiss_sensation"},
    { name: "M_19", color: "#7B221C" , subcategory: "kiss_sensation"},
    { name: "M_20", color: "#B85957" , subcategory: "kiss_sensation"},
    { name: "M_21", color: "#8A3F3C" , subcategory: "kiss_sensation"},
    { name: "M_22", color: "#913B39" , subcategory: "kiss_sensation"},
    { name: "M_23", color: "#6D2B1C" , subcategory: "kiss_sensation"},
    { name: "M_24", color: "#5D202A" , subcategory: "kiss_sensation"},
  ];

  const glossyShades =  [
    { name: "M_01_700415", color: "#700415" },
  { name: "M_02_81071C", color: "#81071C" },
  { name: "M_03_8F131C", color: "#8F131C" },
  { name: "M_04_831E26", color: "#831E26" },
  { name: "M_05_66161A", color: "#66161A" },
  { name: "M_06_3C1213", color: "#3C1213" },
  { name: "M_07_401310", color: "#401310" },
  { name: "M_08_6E182E", color: "#6E182E" },
  { name: "M_09_801620", color: "#801620" },
  { name: "M_10_A95B57", color: "#A95B57" },
  { name: "M_11_5D0D10", color: "#5D0D10" },
  { name: "M_12_983445", color: "#983445" },
  { name: "M_13_60262F", color: "#60262F" },
  { name: "M_14_6E263D", color: "#6E263D" },
  { name: "M_15_812937", color: "#812937" },
  { name: "M_16_BF4951", color: "#BF4951" },
  { name: "M_17_5F2E23", color: "#5F2E23" },
  { name: "M_18_481F1B", color: "#481F1B" },
  { name: "M_19_7B221C", color: "#7B221C" },
  { name: "M_20_B85957", color: "#B85957" },
  { name: "M_21_8A3F3C", color: "#8A3F3C" },
  { name: "M_22_913B39", color: "#913B39" },
  { name: "M_23_6D2B1C", color: "#6D2B1C" },
  { name: "M_24_5D202A", color: "#5D202A" },
  ]; // Same shades for glossy
  // Same shades for glossy

  // ✅ NEW: Lip Liner Shades
  const lipLinerShades = [
    { name: "L_01_5E1F1F", color: "#5E1F1F" }, // Dark Cocoa
    { name: "L_02_431616", color: "#431616" }, // Espresso Brown
    { name: "L_03_7B221C", color: "#7B221C" }, // Deep Wine
    { name: "L_04_6D2B1C", color: "#6D2B1C" }, // Mahogany
    { name: "L_05_913B39", color: "#913B39" }, // Rich Burgundy
  ];

  const eyelinerShades = [
    { name: "E_01_ObsidianBlack", color: "#0A0A0A" },  // Pure Black
    { name: "E_02_EspressoBrown", color: "#3B2F2F" },  // Deep Brown
    { name: "E_03_ChocolateTruffle", color: "#4E2A2A" },  // Rich Chocolate Brown
    { name: "E_04_MidnightCharcoal", color: "#2E2E2E" },  // Dark Charcoal
    { name: "E_05_DeepMocha", color: "#3F2B1F" },  // Warm Mocha Brown
    { name: "E_06_Onyx", color: "#171717" },  // Deep Black Onyx
    { name: "E_07_MahoganyMist", color: "#3A1C1A" },  // Dark Reddish Brown
    { name: "E_08_SableBrown", color: "#463C2E" },  // Cool-Toned Dark Brown
    { name: "E_09_VelvetAubergine", color: "#301934" },  // Almost-Black Plum
    { name: "E_10_Graphite", color: "#2C2C2C" },  // Deep Grey-Black
];

  const eyeshadowShades = [
    { name: "E_01_DarkBerry", color: "#6B284F" },  
    { name: "E_03_DeepCrimson", color: "#7D2B2B" },  
    { name: "M_01_BloodWine", color: "#52030F" }, 
    { name: "M_10_DarkRust", color: "#7D3A35" },  
    { name: "B_02_MidnightSapphire", color: "#243D73" },  
    { name: "G_01_Evergreen", color: "#1E4A2D" },  
    { name: "G_03_OliveNoir", color: "#4A652E" },  
    { name: "Y_03_AntiqueGold", color: "#A67000" },  
    { name: "O_02_BurntSienna", color: "#B54F1C" },  
    { name: "O_03_DeepTerracotta", color: "#D66A32" },  
];


 const hairShades = [
  { name: "H_Black", color: "#1A1A1A" },     // Natural Black
  { name: "H_DarkBrown", color: "#4B2E1D" }, // Dark Brown
  { name: "H_Chestnut", color: "#7B3F00" },  // Medium Chestnut Brown
  { name: "H_Auburn", color: "#8B2500" },    // Reddish Brown (Auburn)
  { name: "H_Burgundy", color: "#800020" },  // Deep Burgundy Red
  { name: "H_Red", color: "#B22222" }        // Natural Red
];


switch (type) {
  case "lipstick":
    return { matteShades, glossyShades, endpoint: "lips" };
  case "lip_liner":
    return { shades: lipLinerShades, endpoint: "lip_liner" };
  case "eyeliner":
    return { shades: eyelinerShades, endpoint: "eyeliner" };
  case "eyeshadow":
    return { shades: eyeshadowShades, endpoint: "eyeshadow" };
  case "concealer":
    return { shades: concealerShades,endpoint: "concealer" };
  case "eyebrows": // ✅ Added eyebrow case
    return { shades: eyebrowShades, endpoint: "eyebrows" };
  case "hair":
    return { shades: hairShades,endpoint: "hair" };
  default:
    throw new Error("Unexpected type provided.");
}
};
