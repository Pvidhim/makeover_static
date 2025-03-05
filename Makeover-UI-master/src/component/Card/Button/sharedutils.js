export const getShadesAndEndpoint = (type) => {
  if (!["lipstick", "lip_liner", "eyeliner", "eyeshadow"].includes(type)) {
    throw new Error("Invalid parameter: type should be 'lipstick', 'lip_liner', 'eyeliner', or 'eyeshadow'.");
  }

  const matteShades = [
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
  ];

  const glossyShades = [...matteShades]; // Same shades for glossy

  // ✅ NEW: Lip Liner Shades
  const lipLinerShades = [
    { name: "L_01_5E1F1F", color: "#5E1F1F" }, // Dark Cocoa
    { name: "L_02_431616", color: "#431616" }, // Espresso Brown
    { name: "L_03_7B221C", color: "#7B221C" }, // Deep Wine
    { name: "L_04_6D2B1C", color: "#6D2B1C" }, // Mahogany
    { name: "L_05_913B39", color: "#913B39" }, // Rich Burgundy
  ];

  const eyelinerShades = [
    { name: "E_07_DarkPlum", color: "#4B0082" },
    { name: "E_01_0D0D0D", color: "#0D0D0D" }, // Dark Black
    { name: "E_08_DeepTeal", color: "#014B43" },
    { name: "E_09_DarkEmerald", color: "#046307" },
    { name: "E_10_Charcoal", color: "#36454F" },
    { name: "E_11_800020", color: "#800020" },
    { name: "E_12_SmokyQuartz", color: "#433E3F" },
    { name: "E_13_Mahogany", color: "#3A0200" },
    { name: "E_14_000080", color: "#000080" },
    { name: "E_15_DeepWine", color: "#522E3A" },
    { name: "E_16_DarkOlive", color: "#3B5323" },
  ];

  const eyeshadowShades = [
    { name: "E_01_8B3B62", color: "#8B3B62" },
    { name: "E_03_973D3D", color: "#973D3D" },
    { name: "M_01_700415", color: "#700415" },
    { name: "M_10_A95B57", color: "#A95B57" },
    { name: "B_02_3A5F9A", color: "#3A5F9A" },
    { name: "G_01_2F5E3D", color: "#2F5E3D" },  // Deep Forest Green  // Muted Royal Blue
    { name: "G_03_617D45", color: "#617D45" },  // True Orange
    { name: "Y_03_D48A00", color: "#D48A00" },  // Golden Yellow // Burnt Orange
    { name: "O_02_E06D32", color: "#E06D32" },  // Sunset Orange
    { name: "O_03_FF8C42", color: "#FF8C42" },  // Soft Coral Orange
    
];

  switch (type) {
    case "lipstick":
      return { matteShades, glossyShades, endpoint: "lips" };
    case "lip_liner": // ✅ NEW CASE FOR LIP LINER
    return { shades: lipLinerShades, endpoint: "lip_liner" }; 
    case "eyeliner":
      return { shades: eyelinerShades, endpoint: "eyeliner" };
    case "eyeshadow":
      return { shades: eyeshadowShades, endpoint: "eyeshadow" };
    default:
      throw new Error("Unexpected type provided.");
  }
};
